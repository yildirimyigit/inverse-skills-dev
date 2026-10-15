"""ManiSkill environments for the inverse-skills project.

Importing this package registers the environments with gymnasium.
"""

from inverse_skills.envs import stow_cube  # noqa: F401  (registers StowCube-v1)
from inverse_skills.envs import drawer_env  # noqa: F401  (registers SlidingDrawer-v1)
