"""전자저울 기반 probe normal-force 검증.

세 부분으로 나뉜다:

``fv.capture``
    스페이스바 capture mode. 기존 force pipeline 의 **최종 출력만 읽는다.**
``fv.analysis``
    robot 값과 전자저울 ground truth 를 맞추고 지표를 낸다. ROS 를 쓰지 않으므로
    측정한 기계가 아닌 곳에서도 돌아간다.
``fv.plotting``
    논문에 넣을 수 있는 그림을 만든다.
"""
