import argparse
import json
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis"), str(ROOT)]
from data_protocol import build_split, standardize_conditions
from checkpoint_protocol import validate_checkpoint_protocol
from configuration import sampling_defaults
from frame_flow_matching import frame_flow_interpolate, frame_velocity_target, pred_frames_from_velocity
from frame_utils import random_so3, so3_exp_map
from ipa_model_dual import ProteinPeptideFrameDenoiser
from lddt_loss import compute_lddt_loss
from physics_losses import TopologyHarmonicPenalty
from reference_metrics import atom_rmsd, nearest_reference, initial_sample_rate
from tica_core import fit_tica
from train import compute_all_losses


class DataProtocolTests(unittest.TestCase):
    def test_split_matches_trajectory_membership(self):
        replica = np.repeat(np.arange(1, 5), 12500)
        frames = np.tile(np.arange(12500), 4)
        split = build_split(replica, frames, "jnj2113")
        self.assertEqual([len(split[k]) for k in ("train", "validation", "excluded")], [45000, 5000, 0])
        self.assertEqual(set(replica[split["train"]]), {1, 2, 3, 4})
        self.assertTrue(np.all(frames[split["validation"]] >= 11250))
        self.assertEqual(len(np.unique(np.concatenate(list(split.values())))), 50000)
        self.assertEqual(len(split["excluded"]), 0)
        for rep in range(1, 5):
            self.assertEqual(int(np.sum(replica[split["train"]] == rep)), 11250)
            self.assertEqual(int(np.sum(replica[split["validation"]] == rep)), 1250)

    def test_condition_statistics_exclude_other_rows(self):
        x = np.array([[1., 2.], [3., 6.], [1000., -500.]])
        transformed, mean, std = standardize_conditions(x, [0, 1])
        np.testing.assert_allclose(mean, [2, 4])
        np.testing.assert_allclose(std, [1, 2])
        np.testing.assert_allclose(transformed[:2], [[-1, -1], [1, 1]])

    def test_tica_does_not_join_replicas(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(12, 3))
        model, _, info = fit_tica(features, [6, 6], lagtime=1)
        self.assertEqual(info["lagged_pairs"], 10)
        self.assertEqual(info["cross_replica_pairs"], 0)
        whole, _, _ = fit_tica(features, [12], lagtime=1)
        self.assertFalse(np.allclose(model.eigenvalues_, whole.eigenvalues_))

    def test_archived_checkpoint_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_protocol({}, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(validate_checkpoint_protocol({}, {}, True), "archived_unverified")

    def test_sampling_matches_recovered_launcher(self):
        defaults = sampling_defaults(ROOT / "configs/jnj2113_manuscript.json")
        self.assertEqual(defaults["num_steps"], 150)
        self.assertEqual(defaults["cfg_scale"], 3)
        self.assertEqual(defaults["disulfide_weight"], 0)
        self.assertEqual(defaults["disulfide_rot_weight"], 0)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_flow_inversion(self):
        r0, r1 = random_so3((2, 4), "cpu"), random_so3((2, 4), "cpu")
        t0, t1, s = torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.tensor([0.2, 0.7])
        rs, ts = frame_flow_interpolate(r0, t0, r1, t1, s)
        v, w = frame_velocity_target(r0, t0, r1, t1)
        recovered_r, recovered_t = pred_frames_from_velocity(rs, ts, v, w, s)
        torch.testing.assert_close(recovered_t, t0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recovered_r, r0, atol=2e-4, rtol=2e-4)

    def test_rotation_at_zero_has_finite_gradient(self):
        x = torch.zeros(2, 3, requires_grad=True)
        (so3_exp_map(x) * torch.randn(2, 3, 3)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_lddt_batch_order_and_single_sample_agree(self):
        target = torch.tensor([[[0.,0,0],[.2,0,0],[2.,0,0]],
                               [[0.,0,0],[2.,0,0],[.3,0,0]]])
        pred = target.clone(); pred[1, 2, 0] += .4
        batch = compute_lddt_loss(pred, target)
        individual = torch.stack([compute_lddt_loss(pred[i:i+1], target[i:i+1]) for i in range(2)]).mean()
        torch.testing.assert_close(batch, individual)
        torch.testing.assert_close(batch, compute_lddt_loss(pred.flip(0), target.flip(0)))

    def test_lddt_mask_really_excludes_atoms(self):
        target = torch.randn(2, 4, 3) * .1
        pred = target.clone(); mask = torch.tensor([[1,1,0,0], [1,1,0,0]], dtype=torch.bool)
        before = compute_lddt_loss(pred, target, mask)
        pred[:, 2:] += 100
        torch.testing.assert_close(before, compute_lddt_loss(pred, target, mask))

    def test_lddt_empty_mask_has_zero_finite_gradient(self):
        pred = torch.randn(2, 3, 3, requires_grad=True)
        loss = compute_lddt_loss(pred, pred.detach(), torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_equivariant_vector_heads(self):
        model = ProteinPeptideFrameDenoiser(4, 3, c_s=16, c_z=8, n_layers=2,
            condition_dim=8, n_heads=2, slow_feat_dim=2, dropout=0,
            n_topo_heads=1, equivariant_translations=True).eval()
        for heads in (model.protein_frame_update_t, model.peptide_frame_update_t,
                      model.protein_frame_update_R, model.peptide_frame_update_R):
            for head in heads:
                torch.nn.init.normal_(head.weight, std=.03)
        pr, er = random_so3((2, 4), "cpu"), random_so3((2, 3), "cpu")
        pt, et = torch.randn(2, 4, 3), torch.randn(2, 3, 3)
        q, shift = random_so3((), "cpu"), torch.tensor([1., -2., .4])
        time, cond = torch.tensor([.2, .7]), torch.randn(2, 2)
        with torch.no_grad():
            baseline = model(pr, pt, er, et, time, slow_feat=cond)
            transformed = model(q @ pr, pt @ q.T + shift, q @ er, et @ q.T + shift, time, slow_feat=cond)
        for i in (0, 2):
            torch.testing.assert_close(transformed[i], baseline[i] @ q.T, rtol=2e-4, atol=2e-4)
        for i in (1, 3):
            torch.testing.assert_close(transformed[i], baseline[i], rtol=2e-4, atol=2e-4)

    def test_auxiliary_weighting_is_per_sample(self):
        b, np_, ne = 2, 4, 3
        pr, er = random_so3((b, np_), "cpu"), random_so3((b, ne), "cpu")
        pt, et = torch.randn(b, np_, 3), torch.randn(b, ne, 3)
        pv, po, ev, eo = [torch.randn_like(x, requires_grad=True) for x in (pt, pt, et, et)]
        s = torch.tensor([.1, .9])
        cfg = json.loads((ROOT / "configs/jnj2113_manuscript.json").read_text())
        args = argparse.Namespace(**cfg["training"]["loss_weights"], sigma_min=1e-4, pair_frame_sep=6)
        args.lambda_disulfide = 0
        nloc = torch.tensor([-.05,.13,0.])
        oloc = torch.tensor([.2,.1,0.])
        meta = {"global_scale":1., "complex_com":torch.zeros(3),
            "protein_N_local":nloc.repeat(np_,1), "protein_O_local":oloc.repeat(np_,1),
            "peptide_N_local":nloc.repeat(ne,1), "peptide_O_local":oloc.repeat(ne,1)}
        tp = TopologyHarmonicPenalty(meta["protein_N_local"],meta["protein_O_local"],1.,torch.zeros(3))
        te = TopologyHarmonicPenalty(meta["peptide_N_local"],meta["peptide_O_local"],1.,torch.zeros(3))
        zeros_p, zeros_e = torch.zeros_like(pt), torch.zeros_like(et)
        inputs = [pv,po,ev,eo,zeros_p,zeros_p,zeros_e,zeros_e,pr,pt,er,et,pr,pt,er,et,s]
        batch, components = compute_all_losses(*inputs,args,meta,tp,te,"cpu")
        separate = [compute_all_losses(*[x[i:i+1] for x in inputs],args,meta,tp,te,"cpu")[0] for i in range(b)]
        torch.testing.assert_close(batch, torch.stack(separate).mean(), atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(components["aux_raw"], components["aux_coord"])
        batch.backward()
        for x in (pv,po,ev,eo):
            self.assertTrue(torch.isfinite(x.grad).all())


class ReferenceMetricTests(unittest.TestCase):
    def test_rmsd_has_euclidean_per_atom_units(self):
        self.assertEqual(atom_rmsd(np.zeros((3,3)), np.ones((3,3))), np.sqrt(3))

    def test_peptide_distance_does_not_refit_peptide(self):
        query = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[2,2,2]])
        reference = query.copy(); reference[-1] += [1,0,0]
        distance, index = nearest_reference(query, reference[None], [0,1,2,3], [4], [0])
        self.assertAlmostEqual(distance, 1., places=10)
        self.assertEqual(index, 0)

    def test_only_explicit_reference_membership_is_used(self):
        q = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[2,2,2]])
        ref = np.stack([q,q]); ref[1,-1] += [2,0,0]
        distance, index = nearest_reference(q, ref, [0,1,2,3], [4], [1])
        self.assertEqual(index, 1); self.assertAlmostEqual(distance, 2.)

    def test_missing_outputs_remain_in_denominator(self):
        self.assertEqual(initial_sample_rate(98, 100), 98.)
        with self.assertRaises(ValueError): initial_sample_rate(101, 100)


if __name__ == "__main__":
    unittest.main()
