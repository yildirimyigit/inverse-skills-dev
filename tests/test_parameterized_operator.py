from inverse_skills.operators import OperatorExtractionConfig, OperatorExtractor, OperatorParameterizer, RoleBindingInferer
from inverse_skills.toy import build_predicate_registry, make_push_rollouts_executable_named, build_predicate_registry_with_distractor, make_push_rollouts_executable_named_with_distractor


def test_role_binding_infers_object_and_regions() -> None:
    rollouts = make_push_rollouts_executable_named(
        object_name="mug",
        source_name="home",
        target_name="goal",
        skill_name="push_restore",
        num_rollouts=1,
    )
    bindings = RoleBindingInferer().infer(rollouts[0])
    assert bindings == {"mug": "?obj", "home": "?src", "goal": "?dst"}


def test_parameterized_template_matches_across_renamings() -> None:
    cases = [
        ("cube", "source", "target"),
        ("mug", "home", "goal"),
    ]
    templates = []
    for object_name, source_name, target_name in cases:
        rollouts = make_push_rollouts_executable_named(
            object_name=object_name,
            source_name=source_name,
            target_name=target_name,
            skill_name="push_restore",
            num_rollouts=2,
        )
        registry = build_predicate_registry(object_name, source_name, target_name)
        learned = OperatorExtractor(registry).extract("push_restore", rollouts)
        template = OperatorParameterizer().parameterize(learned.operator, rollouts[0]).to_dict()
        templates.append(template)

    assert templates[0]["preconditions"] == templates[1]["preconditions"]
    assert templates[0]["add_effects"] == templates[1]["add_effects"]
    assert templates[0]["delete_effects"] == templates[1]["delete_effects"]
    assert templates[0]["inverse_target_terms"] == templates[1]["inverse_target_terms"]


def test_role_binding_ignores_static_distractor() -> None:
    rollouts = make_push_rollouts_executable_named_with_distractor(
        object_name="mug",
        source_name="home",
        target_name="goal",
        distractor_name="bottle",
        skill_name="push_restore",
        num_rollouts=1,
    )
    bindings = RoleBindingInferer().infer(rollouts[0])
    assert bindings == {"mug": "?obj", "home": "?src", "goal": "?dst"}


def _distractor_templates(config=None) -> list[dict]:
    cases = [
        ("cube", "source", "target", "can"),
        ("mug", "home", "goal", "bottle"),
    ]
    templates = []
    for object_name, source_name, target_name, distractor_name in cases:
        rollouts = make_push_rollouts_executable_named_with_distractor(
            object_name=object_name,
            source_name=source_name,
            target_name=target_name,
            distractor_name=distractor_name,
            skill_name="push_restore",
            num_rollouts=2,
        )
        registry = build_predicate_registry_with_distractor(
            object_name=object_name,
            source_name=source_name,
            target_name=target_name,
            distractor_name=distractor_name,
        )
        learned = OperatorExtractor(registry, config).extract("push_restore", rollouts)
        template = OperatorParameterizer().parameterize(learned.operator, rollouts[0]).to_dict()
        templates.append(template)
    return templates


def test_parameterized_template_matches_with_distractor_present() -> None:
    """The workshop paper's claim, under the rule it was made with (the published mean rule, as
    planrob_bundle.py names it): a static distractor leaves the template unchanged."""
    templates = _distractor_templates(OperatorExtractionConfig(mode="mean"))

    assert templates[0]["preconditions"] == templates[1]["preconditions"]
    assert templates[0]["add_effects"] == templates[1]["add_effects"]
    assert templates[0]["delete_effects"] == templates[1]["delete_effects"]
    assert templates[0]["inverse_target_terms"] == templates[1]["inverse_target_terms"]


def test_signed_templates_also_keep_what_is_reliably_false_about_the_distractor() -> None:
    """Superseded by the signed rule (2026-11-11): every reliably observed initial condition is
    kept, false ones included, and no predicate is filtered as irrelevant. The distractor is not
    bound to a role, so its false facts keep its own name, and two scenes with differently named
    distractors give templates that differ exactly there. On everything about the bound roles
    they agree."""
    templates = _distractor_templates()

    def names_an_unbound_object(key: str) -> bool:
        inner = key[key.index("(") + 1:key.rindex(")")] if "(" in key else ""
        return any(arg and not arg.startswith("?") for arg in inner.split(","))

    def split(terms):
        concrete = [t for t in terms if names_an_unbound_object(t["key"])]
        roles = [t for t in terms if t not in concrete]
        return roles, concrete

    for field in ("preconditions", "inverse_target_terms"):
        (roles_0, concrete_0), (roles_1, concrete_1) = split(templates[0][field]), split(templates[1][field])
        assert roles_0 == roles_1, field
        assert {(t["key"], t["polarity"]) for t in concrete_0} == {
            ("holding(can)", "negative"), ("in_region(can,?dst)", "negative"), ("in_region(can,?src)", "negative")}
        assert {(t["key"], t["polarity"]) for t in concrete_1} == {
            ("holding(bottle)", "negative"), ("in_region(bottle,?dst)", "negative"),
            ("in_region(bottle,?src)", "negative")}
    assert templates[0]["add_effects"] == templates[1]["add_effects"]
    assert templates[0]["delete_effects"] == templates[1]["delete_effects"]
