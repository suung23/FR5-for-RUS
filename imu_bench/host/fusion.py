"""자세 추정 — Madgwick AHRS 와 쿼터니언 유틸.

기준 프레임은 Madgwick 원논문과 같은 **NWU** (X=북, Y=서, Z=위) 지구고정 프레임이다.
BNO085 의 rotation vector 는 ENU 를 쓰므로 두 결과를 비교할 땐 고정 오프셋(정확히
Z축 90도)이 남는다 — 뷰어는 그 오프셋을 한 번 재두고 이후 벌어짐만 본다.
"""

import math

import numpy as np


class Madgwick:
    """Madgwick AHRS. mag 이 주어지면 MARG(9축), 아니면 IMU-only(6축) 갱신식을 쓴다.

    상태 q 는 [w, x, y, z] 단위 쿼터니언이며, R(q).T 가 지구->센서 변환이 되도록
    정의된다 (원논문의 ``{}^S_E\\hat{q}`` 규약).
    """

    def __init__(self, beta=0.05):
        self.beta = beta          # gradient-descent gain: 클수록 가속/자력계를 빨리 믿는다
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyr, acc, mag, dt):
        q1, q2, q3, q4 = self.q
        gx, gy, gz = gyr                         # rad/s
        # 자이로 적분항 (쿼터니언 미분)
        qDot = 0.5 * np.array([
            -q2 * gx - q3 * gy - q4 * gz,
             q1 * gx + q3 * gz - q4 * gy,
             q1 * gy - q2 * gz + q4 * gx,
             q1 * gz + q2 * gy - q3 * gx,
        ])

        a_norm = np.linalg.norm(acc)
        if a_norm > 1e-9:
            ax, ay, az = acc / a_norm
            use_mag = mag is not None and np.linalg.norm(mag) > 1e-9
            if use_mag:
                mx, my, mz = np.asarray(mag) / np.linalg.norm(mag)
                # 지구 자기장을 현재 자세로 회전시켜 수평(bx)/수직(bz) 성분으로 분해
                hx = 2*mx*(0.5 - q3*q3 - q4*q4) + 2*my*(q2*q3 - q1*q4) + 2*mz*(q2*q4 + q1*q3)
                hy = 2*mx*(q2*q3 + q1*q4) + 2*my*(0.5 - q2*q2 - q4*q4) + 2*mz*(q3*q4 - q1*q2)
                bx = math.sqrt(hx*hx + hy*hy)
                bz = 2*mx*(q2*q4 - q1*q3) + 2*my*(q3*q4 + q1*q2) + 2*mz*(0.5 - q2*q2 - q3*q3)

                f = np.array([
                    2*(q2*q4 - q1*q3) - ax,
                    2*(q1*q2 + q3*q4) - ay,
                    2*(0.5 - q2*q2 - q3*q3) - az,
                    2*bx*(0.5 - q3*q3 - q4*q4) + 2*bz*(q2*q4 - q1*q3) - mx,
                    2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4) - my,
                    2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2*q2 - q3*q3) - mz,
                ])
                J = np.array([
                    [-2*q3,  2*q4, -2*q1,  2*q2],
                    [ 2*q2,  2*q1,  2*q4,  2*q3],
                    [ 0.0,  -4*q2, -4*q3,  0.0],
                    [-2*bz*q3,           2*bz*q4,           -4*bx*q3 - 2*bz*q1, -4*bx*q4 + 2*bz*q2],
                    [-2*bx*q4 + 2*bz*q2, 2*bx*q3 + 2*bz*q1,  2*bx*q2 + 2*bz*q4, -2*bx*q1 + 2*bz*q3],
                    [ 2*bx*q3,           2*bx*q4 - 4*bz*q2,  2*bx*q1 - 4*bz*q3,  2*bx*q2],
                ])
            else:
                f = np.array([
                    2*(q2*q4 - q1*q3) - ax,
                    2*(q1*q2 + q3*q4) - ay,
                    2*(0.5 - q2*q2 - q3*q3) - az,
                ])
                J = np.array([
                    [-2*q3,  2*q4, -2*q1,  2*q2],
                    [ 2*q2,  2*q1,  2*q4,  2*q3],
                    [ 0.0,  -4*q2, -4*q3,  0.0],
                ])

            grad = J.T @ f
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-9:
                qDot -= self.beta * (grad / gnorm)

        q = self.q + qDot * dt
        self.q = q / np.linalg.norm(q)
        return self.q


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_matrix(q):
    """[w,x,y,z] -> 3x3 회전행렬 (센서 -> 지구)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_from_matrix(R):
    """회전행렬 -> [w,x,y,z]. Shepperd 방식: 가장 큰 대각 성분을 축으로 삼아
    수치적으로 불안정한 경우를 피한다."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        r = math.sqrt(1.0 + t)
        q = np.array([r, (R[2, 1] - R[1, 2]) / r,
                         (R[0, 2] - R[2, 0]) / r,
                         (R[1, 0] - R[0, 1]) / r])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / r
        q[i + 1] = r
        q[j + 1] = (R[j, i] + R[i, j]) / r
        q[k + 1] = (R[k, i] + R[i, k]) / r
    return q / np.linalg.norm(q)


def seed_quat(acc, mag):
    """정지 상태의 가속도+자력계 한 쌍에서 절대 자세를 직접 푼다 (TRIAD).

    Madgwick 을 항등 쿼터니언에서 출발시키면 수렴에 수 초가 걸리고, 그 과도상태는
    실제 드리프트와 구분되지 않는다. 첫 샘플에서 자세를 바로 세워두면 그 구간이
    사라진다. 실패하면(입력이 퇴화) None.
    """
    up = np.asarray(acc, dtype=float)
    n = np.linalg.norm(up)
    if n < 1e-9:
        return None
    up /= n

    # 지구 프레임은 NWU (X=북, Y=서, Z=위) 우수좌표계다. 자기장은 북쪽 성분과
    # 수직 성분만 가지므로 Z x m 은 수직 성분을 죽이고 Z x X = Y, 즉 서쪽을 준다.
    west = np.cross(up, np.asarray(mag, dtype=float))
    n = np.linalg.norm(west)
    if n < 1e-9:          # 자기장이 중력과 평행 — 방위를 정할 수 없다
        return None
    west /= n

    north = np.cross(west, up)   # Y x Z = X

    # Madgwick 의 q 는 R(q).T 가 지구->센서가 되도록 정의된다. 지구 기저벡터를
    # 센서 좌표로 적은 것이 그 열이므로, R(q) 는 그것들을 행으로 갖는다.
    return quat_from_matrix(np.vstack([north, west, up]))


def quat_to_euler_deg(q):
    """[w,x,y,z] -> (roll, pitch, yaw) 도 단위, ZYX 순서."""
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_angle_deg(qa, qb):
    """두 쿼터니언 사이의 회전각(도). 부호 모호성은 |dot| 으로 흡수."""
    d = abs(float(np.dot(qa, qb)))
    return math.degrees(2 * math.acos(max(-1.0, min(1.0, d))))
