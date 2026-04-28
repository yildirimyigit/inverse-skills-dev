from inverse_skills.operators import OperatorExtractor, OperatorParameterizer, RoleBindingInferer
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


def test_parameterized_template_matches_with_distractor_present() -> None:
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
        learned = OperatorExtractor(registry).extract("push_restore", rollouts)
        template = OperatorParameterizer().parameterize(learned.operator, rollouts[0]).to_dict()
        templates.append(template)

    assert templates[0]["preconditions"] == templates[1]["preconditions"]
    assert templates[0]["add_effects"] == templates[1]["add_effects"]
    assert templates[0]["delete_effects"] == templates[1]["delete_effects"]
    assert templates[0]["inverse_target_terms"] == templates[1]["inverse_target_terms"]
