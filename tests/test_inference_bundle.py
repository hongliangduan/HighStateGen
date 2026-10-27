import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis"), str(ROOT / "tools")]
from frame_dataset_dual import ProteinPeptideFrameDataset
from ipa_model_dual import ProteinPeptideFrameDenoiser
from inference_context import collect_context, restore_dataset, restore_geometry, load_context, validate_context
from export_inference_bundle import export_bundle
from physics_losses import build_disulfide_loss
from sample import main as sample_main
from data_protocol import sha256_file

class InferenceBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temp.name)
        cls.top = ROOT / "examples/jnj2113/complex_dry.prmtop"
        import mdtraj as md
        ref = md.load(str(ROOT / "examples/jnj2113/complex_dry.pdb"), top=str(cls.top))
        # Synthetic test frames only: no experimental results are recomputed.
        frames = np.concatenate([ref.xyz, ref.xyz * 1.001 + .01], axis=0).astype(np.float32)
        cls.data = cls.work / "tiny_training.npz"
        np.savez(cls.data, coords=frames, mean=np.zeros(3,np.float32), std=np.ones(3,np.float32))
        cls.stats = cls.work / "stats.npz"
        np.savez(cls.stats, mean=np.array([.2,-.3],np.float32), std=np.array([1.1,.8],np.float32))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.ds = ProteinPeptideFrameDataset(str(cls.data), str(cls.top))
            cls.context = collect_context(cls.ds, cls.top, True)
        cls.model_args = dict(c_s=16,c_z=8,n_layers=2,condition_dim=8,n_heads=2,
            slow_feat_dim=2,n_topo_heads=1,khop=3,cyclic_peptide=True,
            equivariant_translations=False,topology_graph_scope="complex")
        torch.manual_seed(9)
        with contextlib.redirect_stdout(io.StringIO()):
            model = ProteinPeptideFrameDenoiser(
                294,13,topo_path=str(cls.top),peptide_residue_ids=cls.ds.peptide_residue_ids,
                dropout=0,**cls.model_args)
        cls.checkpoint = cls.work / "source.pt"
        torch.save({"args": cls.model_args, "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":{"unneeded":torch.ones(10)}, "epoch":2},cls.checkpoint)
        cls.bundle = cls.work / "bundle"
        export_args = argparse.Namespace(
            checkpoint=str(cls.checkpoint),weights_key="model_state_dict",
            data=str(cls.data),topology=str(cls.top),condition_stats=str(cls.stats),
            sampling_config=str(ROOT/"configs/jnj2113_manuscript.json"),
            model_args=None,tica_dir=None,allow_legacy_checkpoint=True,output=str(cls.bundle))
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            export_bundle(export_args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_statistics_and_geometry_round_trip(self):
        cached = restore_dataset(self.context)
        for key in ("protein_N_local","protein_O_local","peptide_N_local","peptide_O_local","complex_com"):
            torch.testing.assert_close(getattr(cached,key),getattr(self.ds,key),rtol=0,atol=0)
        self.assertEqual(cached.global_scale,self.ds.global_scale)
        self.assertFalse(hasattr(cached,"coords_nm"))
        with contextlib.redirect_stdout(io.StringIO()):
            rebuilt, _ = restore_geometry(self.context,cached)
            direct = build_disulfide_loss(str(self.top),self.ds.peptide_residue_ids,
                self.ds.peptide_N_local,self.ds.global_scale,self.ds.complex_com,
                self.ds.peptide_bb_indices,self.ds.coords_nm)
        self.assertEqual(rebuilt.ref_cbcb_dist,direct.ref_cbcb_dist)
        self.assertEqual(rebuilt.ref_caca_dist,direct.ref_caca_dist)

    def test_rejects_mismatched_inputs(self):
        b=self.bundle
        args=torch.load(b/"inference.pt",weights_only=True)["args"]
        load_context(b/"inference_context.pt",b/"inference.pt",b/"complex_dry.prmtop",b/"condition_stats.npz",args)
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            load_context(b/"inference_context.pt",self.checkpoint,b/"complex_dry.prmtop",b/"condition_stats.npz",args)
        changed=dict(args,translation_scale_mode="radial_rms")
        with self.assertRaisesRegex(ValueError,"coordinate scales"):
            load_context(b/"inference_context.pt",b/"inference.pt",b/"complex_dry.prmtop",b/"condition_stats.npz",changed)
        with self.assertRaisesRegex(ValueError,"normalization"):
            load_context(b/"inference_context.pt",b/"inference.pt",b/"complex_dry.prmtop",None,args)

    def test_small_bundle_has_no_training_frames_or_optimizer(self):
        weights=torch.load(self.bundle/"inference.pt",weights_only=True)
        self.assertNotIn("optimizer_state_dict",weights)
        context=torch.load(self.bundle/"inference_context.pt",weights_only=True)
        self.assertNotIn("coords",context["dataset"])
        self.assertNotIn("coords_nm",context["dataset"])
        self.assertLess((self.bundle/"inference_context.pt").stat().st_size,100000)
        self.assertEqual(context["source"]["training_frame_count"],2)

    def test_invalid_statistics_rejected(self):
        invalid=copy.deepcopy(self.context)
        invalid["dataset"]["global_scale"]=float("nan")
        with self.assertRaisesRegex(ValueError,"coordinate scale"):
            validate_context(invalid)
        invalid=copy.deepcopy(self.context)
        invalid["dataset"]["protein_N_local"][0,0]=float("inf")
        with self.assertRaisesRegex(ValueError,"template"):
            validate_context(invalid)

    def test_sampling_matches_after_training_npz_removed(self):
        common=["sample.py","--n_samples","1","--num_steps","2","--seed","13",
                "--device","cpu","--slow_feat","0.4","-0.2","--allow_legacy_checkpoint"]
        direct=self.work/"direct.pdb"
        cached=self.work/"cached.pdb"
        with warnings.catch_warnings(),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            with patch.object(sys,"argv",common+[
                "--checkpoint",str(self.checkpoint),"--data",str(self.data),
                "--topology",str(self.top),"--slow_feat_stats",str(self.stats),
                "--config",str(ROOT/"configs/jnj2113_manuscript.json"),"--output",str(direct)]):
                sample_main()
            unavailable=self.data.with_suffix(".hidden")
            self.data.rename(unavailable)
            try:
                with patch.object(sys,"argv",common+[
                    "--checkpoint",str(self.bundle/"inference.pt"),
                    "--inference_context",str(self.bundle/"inference_context.pt"),
                    "--output",str(cached)]):
                    sample_main()
            finally:
                unavailable.rename(self.data)
        self.assertEqual(direct.read_bytes(),cached.read_bytes())
        info=json.loads(Path(str(cached)+".sampling.json").read_text())
        self.assertIsNone(info["inference_input_sha256"]["train_data"])
        self.assertEqual(info["inference_input_sha256"]["inference_context"],sha256_file(self.bundle/"inference_context.pt"))

if __name__=="__main__":
    unittest.main()
