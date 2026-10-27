import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'analysis')]
from configuration import training_defaults, validate_manuscript_arguments
from data_protocol import build_split, sha256_file, verify_manifest
from checkpoint_protocol import validate_checkpoint_protocol
from prepare_training_data import prepare
from frame_utils import random_so3
from ipa_model_dual import ProteinPeptideFrameDenoiser
from physics_losses import TopologyHarmonicPenalty
from train import train_epoch, validate, EMAModel

class TrainingReleaseTests(unittest.TestCase):
    def test_luna_partition_is_unchanged(self):
        rep=np.repeat(np.arange(1,5),12500);frame=np.tile(np.arange(12500),4)
        split=build_split(rep,frame,'luna18')
        self.assertEqual([len(split[k]) for k in ('train','validation','excluded')],[30000,7500,12500])
        self.assertEqual(set(rep[split['excluded']]),{4})

    def test_profile_and_checkpoint_partition(self):
        settings=training_defaults(ROOT/'configs/jnj2113_manuscript.json')
        self.assertEqual(settings['manuscript_protocol'],'jnj2113_45000_v1')
        self.assertEqual(settings['translation_scale_mode'],'coordinate_rms')
        self.assertIs(settings['equivariant_translations'],False)
        self.assertEqual(settings['topology_graph_scope'],'complex')
        record={'split':{'system':'jnj2113','counts':{'train':45000,'validation':5000,'excluded':0},'development_replicas':[1,2,3,4]}}
        self.assertEqual(validate_checkpoint_protocol(settings,record),'manuscript_configured_run')
        for key, bad in [('translation_scale_mode','radial_rms'),('equivariant_translations',True),('topology_graph_scope','peptide')]:
            with self.subTest(setting=key), self.assertRaises(ValueError):
                validate_checkpoint_protocol(dict(settings, **{key:bad}),record)
        record['split']['counts']['train']=30000
        with self.assertRaises(ValueError):validate_checkpoint_protocol(settings,record)

    def test_profile_rejects_conflicting_config_and_cli_settings(self):
        config=json.loads((ROOT/'configs/jnj2113_manuscript.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            for section,key,bad in [('training','translation_scale_mode','radial_rms'),('model','equivariant_translations',True),('model','topology_graph_scope','peptide')]:
                altered=json.loads(json.dumps(config));altered[section][key]=bad
                p=Path(folder)/'config.json';p.write_text(json.dumps(altered))
                with self.subTest(source='config',key=key),self.assertRaises(ValueError):training_defaults(p)
                args=argparse.Namespace(**training_defaults(ROOT/'configs/jnj2113_manuscript.json'))
                setattr(args,key,bad)
                with self.subTest(source='cli',key=key),self.assertRaises(ValueError):validate_manuscript_arguments(args)

    def test_missing_resume_and_extra_epochs_are_rejected(self):
        defaults=training_defaults(ROOT/'configs/jnj2113_manuscript.json')
        args=argparse.Namespace(**defaults)
        args.epochs=1;args.seed=42;args.resume='missing_checkpoint.pt'
        args.val_data='val.npz';args.split_manifest='split.json'
        args.slow_feat_path='train.npy';args.slow_feat_val_path='val.npy';args.train_basin_labels='labels.npy'
        args.disable_tica_conditioning=False;args.disable_topology_bias=False;args.disable_interface_attention=False
        with self.assertRaises(FileNotFoundError):validate_manuscript_arguments(args)
        args.resume=None;args.finetune_extra_epochs=1
        with self.assertRaises(ValueError):validate_manuscript_arguments(args)

    def test_prepare_and_manifest_bind_45000_actual_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);rng=np.random.default_rng(8)
            coords=rng.normal(size=(50000,2,3)).astype(np.float32)
            arrays={'coords_nm':coords,'replica_ids':np.repeat(np.arange(1,5),12500),
                    'frame_indices':np.tile(np.arange(12500),4),'tica_scores':rng.normal(size=(50000,2)),
                    'state_centers':rng.normal(size=(5,2))}
            for k,v in arrays.items():np.save(p/(k+'.npy'),v)
            metadata={'reduction_method':'tica','lagtime':5,'n_align_ca':294,'n_feature_ca':307,'n_frames':50000,
              'tica_fit_details':{'trajectory_lengths':[12500]*4,'cross_replica_pairs':0},
              'fixed_artifact_sha256':{'projection_scores.npy':sha256_file(p/'tica_scores.npy'),'state_centers.npy':sha256_file(p/'state_centers.npy')}}
            (p/'tica_metadata.json').write_text(json.dumps(metadata))
            args=argparse.Namespace(system='jnj2113',output_dir=str(p/'prepared'),tica_metadata=str(p/'tica_metadata.json'),**{k:str(p/(k+'.npy')) for k in arrays})
            with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            out=p/'prepared';inputs={'train_data':out/'train_set.npz','validation_data':out/'val_set.npz',
                'train_conditions':out/'condition_train_normalized.npy','validation_conditions':out/'condition_validation_normalized.npy','train_labels':out/'basin_labels_train.npy'}
            record=verify_manifest(out/'split_manifest.json',inputs)
            self.assertEqual(record['counts'],{'train':45000,'validation':5000,'excluded':0})
            with np.load(out/'train_set.npz') as data:
                expected=np.concatenate([np.arange(r*12500,r*12500+11250) for r in range(4)])
                np.testing.assert_array_equal(data['source_rows'],expected)
                np.testing.assert_array_equal(data['coords'],coords[expected])
            self.assertFalse((out/'excluded_set.npz').exists())
            self.assertEqual(json.loads((out/'split_manifest.json').read_text())['condition_statistics_scope'],'45000 training frames only')

    def test_one_training_step_validation_and_weights_round_trip(self):
        torch.manual_seed(18);torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            model=ProteinPeptideFrameDenoiser(4,3,c_s=16,c_z=8,n_layers=2,condition_dim=8,n_heads=2,slow_feat_dim=2,dropout=0,n_topo_heads=1,equivariant_translations=False,topology_graph_scope='complex')
            defaults=training_defaults(ROOT/'configs/jnj2113_manuscript.json')
            defaults.update(bf16=False,output_dir=folder,pair_frame_sep=3)
            args=argparse.Namespace(**defaults);args.lambda_disulfide=0
            n=torch.tensor([-.05,.13,0.]);o=torch.tensor([.2,.1,0.])
            meta={'global_scale':1.,'complex_com':torch.zeros(3),'protein_N_local':n.repeat(4,1),'protein_O_local':o.repeat(4,1),'peptide_N_local':n.repeat(3,1),'peptide_O_local':o.repeat(3,1)}
            prot=TopologyHarmonicPenalty(meta['protein_N_local'],meta['protein_O_local'],1.,torch.zeros(3))
            pep=TopologyHarmonicPenalty(meta['peptide_N_local'],meta['peptide_O_local'],1.,torch.zeros(3))
            batch={'protein_R':random_so3((2,4),'cpu'),'protein_t':torch.randn(2,4,3),'peptide_R':random_so3((2,3),'cpu'),'peptide_t':torch.randn(2,3,3),'protein_aa':torch.zeros(2,4,dtype=torch.long),'peptide_aa':torch.zeros(2,3,dtype=torch.long),'slow_feat':torch.randn(2,2)}
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            scaler=torch.amp.GradScaler('cpu',enabled=False);ema=EMAModel(model,decay=.99)
            before={k:v.clone() for k,v in model.state_dict().items()}
            metrics=train_epoch(model,[batch],optimizer,scaler,'cpu',args,0,scheduler,ema,meta,prot,pep)
            self.assertGreater(metrics['loss'],0);self.assertTrue(all(np.isfinite(list(metrics.values()))))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
            val=validate(model,[batch],'cpu',args,meta,prot,pep)
            self.assertTrue(all(np.isfinite(list(val.values()))))
            checkpoint=Path(folder)/'step.pt';torch.save({'model_state_dict':model.state_dict()},checkpoint)
            restored=torch.load(checkpoint,weights_only=True)
            model.load_state_dict(restored['model_state_dict'],strict=True)
            self.assertTrue(all(torch.equal(v,restored['model_state_dict'][k]) for k,v in model.state_dict().items()))

if __name__=='__main__':unittest.main()
