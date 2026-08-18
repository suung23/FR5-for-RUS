import PyKDL as kdl
import numpy as np

import math

def build_fairino_chain():
    chain = kdl.Chain()

    # Segment 1: j1 to j2
    chain.addSegment(kdl.Segment(
        kdl.Joint("j1", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))
    ))

    # Segment 2: j2 to j3
    chain.addSegment(kdl.Segment(
        kdl.Joint("j2", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))
    ))

    # Segment 3: j3 to j4
    chain.addSegment(kdl.Segment(
        kdl.Joint("j3", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))
    ))

    # Segment 4: j4 to j5
    chain.addSegment(kdl.Segment(
        kdl.Joint("j4", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))
    ))

    # Segment 5: j5 to j6
    chain.addSegment(kdl.Segment(
        kdl.Joint("j5", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))
    ))

    # Segment 6: j6 to tip
    chain.addSegment(kdl.Segment(
        kdl.Joint("j6", kdl.Joint.RotZ),
        kdl.Frame.Identity()
    ))

    return chain


def calc_T_space_j6(joint_positions):
    chain = build_fairino_chain()
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)

    q = kdl.JntArray(6)
    for i in range(6):
        q[i] = joint_positions[i]

    frame = kdl.Frame()
    fk_solver.JntToCart(q, frame)
    T = np.eye(4)
    for i in range(3):
        for j in range(3):
            T[i, j] = frame.M[i, j]
        T[i, 3] = frame.p[i]
    return T


T_j6_rod_center = np.array([[math.cos(math.radians(25)), -math.sin(math.radians(25)), 0.0, 0.16],
                    [-math.sin(math.radians(25)), -math.cos(math.radians(25)), 0.0, -0.04],
                    [0.0, 0.0, -1, 0.21],
                    [0.0, 0.0, 0.0, 1.0]], dtype=float)


if __name__ == "__main__":
    T_space_j6 = calc_T_space_j6([0.0 ,-1.5708, 1.5708, 0.0, 0.0, 0.0])
    T_space_rod_center = T_space_j6 @ T_j6_rod_center
    print(T_space_rod_center)