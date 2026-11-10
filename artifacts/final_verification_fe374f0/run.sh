#!/bin/bash
# Final verification of fe374f0: the documented Stow and Drawer pipelines, run fresh with the
# minimal planner, the trained policies loaded as they are (nothing retrained).
#
# Every step runs through run_with_provenance.py, which first checks the checkout and the planner
# modules this Python imports, and records every planning call. The inputs of each stage are the
# outputs of the fresh stage before it, never the saved files. Outputs go to this folder only;
# nothing saved is touched, and no step overwrites a file (the scripts refuse to).
#
# The commands are the ones the 2026-11-05 validation used for its plain-script run
# (artifacts/default_mode_validation/after_migration, legacy exact planner), with this folder
# as the output, plus the final targeted-versus-direct evaluation.
#
#   bash artifacts/final_verification_fe374f0/run.sh stage1    # fresh forward operators
#   bash artifacts/final_verification_fe374f0/run.sh smoke     # one seed through Stages 2 and 4, one final-evaluation seed
#   bash artifacts/final_verification_fe374f0/run.sh stage2    # plans and specs, direct spec, Stage 3 input, planner demo
#   bash artifacts/final_verification_fe374f0/run.sh stow4     # Stow Stage 4, seeds 1000-1009
#   bash artifacts/final_verification_fe374f0/run.sh drawer4   # Drawer Stage 4, seeds 2000-2019
#   bash artifacts/final_verification_fe374f0/run.sh final     # targeted versus direct, seeds 3000-3049

set -u
cd /workspace/inverse-skills-dev
export PYTHONNOUSERSITE=1
F=artifacts/final_verification_fe374f0
mkdir -p $F/logs $F/provenance $F/smoke

step() {
  name="$1"; shift
  if [ -e $F/logs/$name.stdout ]; then echo "$name: already run, log kept, not run again"; return; fi
  SECONDS=0
  python $F/run_with_provenance.py $F/provenance/$name.json "$@" > $F/logs/$name.stdout 2>&1
  code=$?
  echo "$name: exit $code in ${SECONDS}s"
  grep -E "STAGE [0-9]:|Traceback|not run:|exists|wrote" $F/logs/$name.stdout | cut -c1-160
}

case "$1" in
  stage1)
    step stow_stage1 scripts/stow_stage1_extract.py --out $F/stow_stage1_operator.json
    step drawer_stage1 scripts/drawer_stage1_extract.py --out $F/drawer_stage1_operator.json
    ;;
  smoke)
    step smoke_drawer_stage2 scripts/drawer_stage2_plan.py --seeds 0 --operator $F/drawer_stage1_operator.json \
      --out $F/smoke/drawer_stage2_plan.json --spec-out $F/smoke/drawer_hole_spec.json
    step smoke_drawer_stage4 scripts/drawer_stage4_execute.py --seeds 2000 --operator $F/drawer_stage1_operator.json \
      --spec $F/smoke/drawer_hole_spec.json --out $F/smoke/drawer_stage4.json
    step smoke_final scripts/drawer_final_evaluate.py --seeds 3000 --workers 2 --operator $F/drawer_stage1_operator.json \
      --out $F/smoke/drawer_final_targeted_vs_direct.json
    ;;
  stage2)
    step stow_stage2 scripts/stow_stage2_plan.py --operator $F/stow_stage1_operator.json \
      --out $F/stow_stage2_plan.json --spec-out $F/stow_hole_spec.json
    step drawer_stage2 scripts/drawer_stage2_plan.py --operator $F/drawer_stage1_operator.json \
      --out $F/drawer_stage2_plan.json --spec-out $F/drawer_hole_spec.json
    step direct_spec scripts/drawer_direct_spec.py --operator $F/drawer_stage1_operator.json \
      --grasp-spec $F/drawer_hole_spec.json --out-dir $F/direct
    step stow_stage3_input scripts/validate_modes_execution.py stage3-input stow --spec $F/stow_hole_spec.json --out-dir $F
    step drawer_stage3_input scripts/validate_modes_execution.py stage3-input drawer --spec $F/drawer_hole_spec.json --out-dir $F
    step stow_planner_demo scripts/stow_planner_demo.py --operator $F/stow_stage1_operator.json --out $F/stow_planner_demo.json
    ;;
  stow4)
    step stow_stage4 scripts/stow_stage4_execute.py artifacts/stow/stage3/1790788534/sac_hole_best \
      --operator $F/stow_stage1_operator.json --spec $F/stow_hole_spec.json \
      --out $F/stow_stage4.json --learned-out $F/stow_learned_operator.json
    ;;
  drawer4)
    step drawer_stage4 scripts/drawer_stage4_execute.py --operator $F/drawer_stage1_operator.json \
      --spec $F/drawer_hole_spec.json --out $F/drawer_stage4.json
    ;;
  final)
    step drawer_final scripts/drawer_final_evaluate.py --operator $F/drawer_stage1_operator.json \
      --out $F/drawer_final_targeted_vs_direct.json
    ;;
  *)
    echo "usage: run.sh stage1|smoke|stage2|stow4|drawer4|final"; exit 2 ;;
esac
