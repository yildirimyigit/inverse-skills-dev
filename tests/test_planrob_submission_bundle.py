from inverse_skills.planrob_bundle import build_bundle


def test_planrob_bundle_core_claims() -> None:
    bundle = build_bundle()
    summary = bundle["summary"]

    assert summary["same_class_inverse_success"] is True
    assert summary["cross_class_inverse_success"] is True
    assert summary["cross_class_inverse_actions"] == ["pick(cube)", "place(source)"]
    assert summary["renaming_template_invariance"] is True
    assert summary["distractor_template_invariance"] is True
    assert summary["pick_place_restoration_gain"] > 0.0
    assert summary["push_restoration_gain"] > 0.0


def test_planrob_bundle_grasp_hold() -> None:
    bundle = build_bundle()
    summary = bundle["summary"]
    grasp = bundle["executable"]["grasp_hold"]

    assert summary["grasp_hold_inverse_success"] is True
    assert summary["grasp_hold_inverse_actions"] == ["place(source)"]
    assert summary["grasp_hold_restoration_gain"] > 0.0

    op = grasp["operator"]
    prec_keys = [t["key"] for t in op["preconditions"]]
    add_keys = [t["key"] for t in op["add_effects"]]
    del_keys = [t["key"] for t in op["delete_effects"]]
    assert "in_region(cube,source)" in prec_keys
    assert "gripper_open()" in prec_keys
    assert "holding(cube)" in add_keys
    assert "gripper_open()" in del_keys

    # grasp_hold has a different operator than pick_place/push (1-step inverse)
    assert len(summary["grasp_hold_inverse_actions"]) == 1
    assert len(summary["cross_class_inverse_actions"]) == 2
