from dataclasses import asdict
from typing import Optional


def _patch_numpy_for_gym() -> None:
    """Gym 旧版が NumPy 2.x で削除された np.bool8 を参照して落ちる問題の暫定回避。

    - np.bool8 が無ければ np.bool_ にエイリアスする。
    """
    try:
        import numpy as _np

        if not hasattr(_np, "bool8"):
            _np.bool8 = _np.bool_  # type: ignore[attr-defined]
    except Exception:
        pass


class _GymToGymnasiumV26:
    """Minimal wrapper to adapt Gym (v0) API to Gymnasium v0.26+ API.

    - reset() -> (obs, info)
    - step(a) -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *, seed=None, options=None):
        if hasattr(self._env, "reset"):
            if seed is not None and hasattr(self._env, "seed"):
                try:
                    self._env.seed(seed)
                except Exception:
                    pass
            obs = self._env.reset()
            return obs, {}
        raise RuntimeError("Underlying env has no reset()")

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


def make_crafter_env(env_cfg) -> "torchrl.envs.EnvBase":  # type: ignore[name-defined]
    """Crafter 環境を TorchRL の `TransformedEnv` として構築する。

    設定 `env_cfg` に応じて以下を適用:
    - ToTensorImage (常時)
    - GrayScale（任意）
    - Resize（任意）
    - CatFrames（フレームスタック任意）
    """
    # 依存
    try:
        _patch_numpy_for_gym()
        import gym  # Crafter は Gym 登録を提供
    except Exception as e:
        raise RuntimeError("Gym is required. `pip install gym==0.26.2`") from e

    try:
        import crafter  # noqa: F401 登録を確実に起こす
    except Exception as e:
        raise RuntimeError("Crafter is required. `pip install crafter`") from e

    try:
        from torchrl.envs import GymWrapper
        from torchrl.envs.transforms import (
            TransformedEnv,
            Compose,
            ToTensorImage,
            GrayScale,
            Resize,
            CatFrames,
            RenameTransform,
        )
    except Exception as e:
        raise RuntimeError("TorchRL is required. `pip install torchrl`") from e

    # Gym 環境生成
    env_id = getattr(env_cfg, "name", "CrafterReward-v1")
    try:
        base_gym = gym.make(env_id)
    except Exception:
        alt = "CrafterNoReward-v1" if env_id == "CrafterReward-v1" else "CrafterReward-v1"
        base_gym = gym.make(alt)

    # Gym の自動ラッパ（TimeLimit 等）を外すことで旧API(4タプル)に揃える
    try:
        base_gym = base_gym.unwrapped
    except Exception:
        pass

    # Gym -> Gymnasium API 変換（簡易アダプタ）
    base_gym = _GymToGymnasiumV26(base_gym)

    # TorchRL の GymWrapper
    base = GymWrapper(base_gym)

    # 観測キーを特定（'pixels' 優先、無ければ 'observation'）
    try:
        keys = set(base.observation_spec.keys(True, True))
    except Exception:
        keys = {"observation"}
    obs_key = "pixels" if "pixels" in keys else "observation"

    # 前処理パイプライン（設定駆動）
    tfs = [ToTensorImage(in_keys=[obs_key])]

    if getattr(env_cfg, "grayscale", True):
        tfs.append(GrayScale(in_keys=[obs_key]))

    img_size = int(getattr(env_cfg, "image_size", 64))
    if img_size and img_size > 0:
        tfs.append(Resize(img_size, img_size, in_keys=[obs_key]))

    frame_stack = int(getattr(env_cfg, "frame_stack", 1))
    if frame_stack and frame_stack > 1:
        # channels-first を前提としてフレーム連結（C,H,W -> C*N,H,W なので dim=-3）
        tfs.append(CatFrames(N=frame_stack, dim=-3, in_keys=[obs_key]))

    # 最後にキーを 'observation' に正規化
    if obs_key != "observation":
        tfs.append(RenameTransform(in_keys=[obs_key], out_keys=["observation"]))

    env = TransformedEnv(base, Compose(*tfs))
    return env

