# Lung Cancer Detection CSP


## Caltech annotation meaning
    % bbGt version=3
    person 227 135 6 12 0 0 0 0 0 0 0

    第一个字段表示类别，包含[preson,people,person-fa,person?]
    第二个字段表示行人全身框的左上角横坐标-X
    第三个字段表示行人全身框的左上角纵坐标-Y
    第四个字段表示行人全身框的宽-W
    第五个字段表示行人全身框的高-H
    第六个字段表示行人的遮挡状况 occ-flag，为0表示行人未被遮挡，可视框为[0,0,0,0]，为1表示行人被遮挡，对应行人的可视框标注
    第七个字段表示行人的可视框的左上角横坐标-X
    第八个字段表示行人的可视框的左上角纵坐标-Y
    第九个字段表示行人可视框的宽-W
    第十个字段表示行人可视框的高-H
    第十一个字段表示该行人的ignore-flag，为0表示行人，为1表示ignore

## PIP China Source
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple

## pre-commit hook install
pre-commit install