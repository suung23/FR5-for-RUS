"""세션 로깅 — 퓨전 스텝마다 한 행씩 CSV 로 남기고, 메타데이터는 사이드카 JSON 에.

CSV 를 기본으로 삼은 이유: 250Hz 로 10분을 받아도 수십 MB 수준이고, 아무 도구로나
바로 열어볼 수 있다. HDF5 는 --format hdf5 로 선택한다 (SurgiTagV2 데이터셋과 같은
방식으로 후처리할 때).

절대시간(Unix epoch, UTC)과 디바이스 시간(µs)을 둘 다 남긴다 — 다른 소스와 병합할
땐 절대시간이, 샘플 간격을 따질 땐 디바이스 시간이 기준이 된다.
"""

import csv
import json
import os
import time

import numpy as np

COLUMNS = [
    "pc_unix",        # PC 절대시각 (Unix epoch, UTC 초)
    "dev_us",         # 디바이스 이벤트 시각 (µs, wrap 펼침 완료)
    "ax", "ay", "az",             # 가속도 [m/s^2]
    "gx", "gy", "gz",             # 자이로 [rad/s]
    "mx_raw", "my_raw", "mz_raw",  # 자력계 원시 [µT]
    "mx", "my", "mz",              # 자력계 보정 후 [µT]
    "host_qw", "host_qx", "host_qy", "host_qz",   # 호스트 Madgwick 퓨전
    "host_roll", "host_pitch", "host_yaw",
    "chip_qw", "chip_qx", "chip_qy", "chip_qz",   # BNO085 내장 퓨전 (RV)
    "chip_roll", "chip_pitch", "chip_yaw",
    "diff_deg",       # 프레임 정렬 후 호스트 vs 칩 각도차
    # 영점(zero calibration) 기준 상대 자세 — "처음 자세에서 얼마나 돌았는가".
    # 영점을 안 잡았으면 빈 값이다.
    "rel_roll", "rel_pitch", "rel_yaw",
    # BNO085 보정 정확도 (0~3). **끝에 붙인다** — 이 CSV 를 이름이 아니라 위치로
    # 읽는 소비자가 생겨도 앞쪽이 안 밀리도록.
    "cal_acc", "cal_gyr", "cal_mag", "cal_rv",
]


def _stamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class SessionLogger:
    """열려 있는 동안 append() 로 한 행씩 받아 쓴다. with 문으로 쓰는 걸 권장."""

    def __init__(self, out_dir, fmt="csv", meta=None, prefix="imu"):
        os.makedirs(out_dir, exist_ok=True)
        self.fmt = fmt
        self.base = os.path.join(out_dir, "%s_%s" % (prefix, _stamp()))
        self.path = self.base + (".csv" if fmt == "csv" else ".h5")
        self.n = 0
        self._start = time.time()
        self._meta = dict(meta or {})

        if fmt == "csv":
            self._fh = open(self.path, "w", newline="")
            self._w = csv.writer(self._fh)
            self._w.writerow(COLUMNS)
        elif fmt == "hdf5":
            import h5py
            self._fh = h5py.File(self.path, "w")
            self._ds = self._fh.create_dataset(
                "samples", shape=(0, len(COLUMNS)), maxshape=(None, len(COLUMNS)),
                dtype="f8", chunks=(1024, len(COLUMNS)), compression="gzip")
            self._ds.attrs["columns"] = COLUMNS
            self._buf = []
        else:
            raise ValueError("unknown log format: %s" % fmt)

        # 메타를 **여는 즉시** 한 번 쓴다. close() 때만 쓰면 프로세스가 강제 종료되거나
        # 장치가 빠져 세션이 끊길 때 영점(zero_ref)이 통째로 사라진다 — CSV 는 남는데
        # 그걸 해석할 기준이 없어지는, 사후 복구가 불가능한 손실이다.
        self._write_meta(final=False)

    def append(self, row):
        if self.fmt == "csv":
            self._w.writerow(["" if v is None else v for v in row])
        else:
            self._buf.append([np.nan if v is None else v for v in row])
            if len(self._buf) >= 1024:
                self._flush_hdf5()
        self.n += 1

    def _flush_hdf5(self):
        if not self._buf:
            return
        block = np.asarray(self._buf, dtype="f8")
        self._ds.resize(self._ds.shape[0] + len(block), axis=0)
        self._ds[-len(block):] = block
        self._buf = []

    def _meta_dict(self, final):
        meta = dict(self._meta)
        meta.update({
            "n_samples": self.n,
            "start_unix": self._start,
            "columns": COLUMNS,
            "closed": bool(final),
        })
        if final:
            meta["end_unix"] = time.time()
            meta["duration_s"] = round(time.time() - self._start, 3)
        return meta

    def _write_meta(self, final):
        """CSV 는 사이드카 JSON 에 원자적으로 쓴다 (임시파일 -> rename)."""
        if self.fmt != "csv":
            return
        path = self.base + ".meta.json"
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._meta_dict(final), f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    def close(self):
        meta = self._meta_dict(final=True)
        if self.fmt == "csv":
            self._fh.close()
            self._write_meta(final=True)
        else:
            self._flush_hdf5()
            for k, v in meta.items():
                try:
                    self._fh.attrs[k] = v
                except TypeError:          # dict 등 h5 속성으로 못 넣는 값
                    self._fh.attrs[k] = json.dumps(v, ensure_ascii=False, default=str)
            self._fh.close()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
