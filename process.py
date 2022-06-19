import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
# import torch
# from crnn.lib import dataset, utils
# from crnn.Chinese_alphabet import alphabet

# converter = utils.strLabelConverter(alphabet, ignore_case=False)
# 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列，使用英文字典时忽略大小写

# 读取并转换图像大小为100 x 32 w x h
# 图像大小转换器
# transformer = dataset.resizeNormalize((100, 32))
videoCount = [0, 0]
imgCount = [0, 0]


def crnnPaddle(result):
    pred, confidence = result
    try:
        computing, outcome = pred.split('=')
        if eval(computing) == eval(outcome):
            return 1
        else:
            return 0
    except Exception as e:
        print(e)
        return 0


# def crnnPredict(img, model):
#     image = Image.fromarray(img).convert('L')
#     image = transformer(image)
#     # print(image)
#     # cv2.imshow('temp', image)
#     # cv2.waitKey(0)
#     if torch.cuda.is_available():
#         image = image.cuda()
#     image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)
#     # print(image.size())
#
#     with torch.no_grad():
#         model.eval()
#         preds = model(image)  # (w c nclass) (26, 1, 37) 26为ctc生成路径长度也是传入rnn的时间步长，1是batchsize，37是字符类别数
#
#     _, preds = preds.max(2)  # 取可能性最大的indecis size (26, 1)
#     preds = preds.transpose(1, 0).contiguous().view(-1)  # 转成以为索引列表
#     # 转成字符序列
#     preds_size = torch.IntTensor([preds.size(0)])
#     # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     # print('%-20s => %-20s' % (raw_pred, sim_pred))
#     # print(sim_pred)
#     sim_pred = sim_pred.replace('×', '*')
#     sim_pred = sim_pred.replace('÷', '/')
#     predComputing, predResult = sim_pred.split('=')
#     return 1 if eval(predComputing) == eval(predResult) else 0


def crnnCheck(pred: str) -> int:
    pred = pred.replace('×', '*')
    pred, predResult = pred.split('=')
    pred = ' '.join(pred)
    """
    这是中缀表达式求值的函数
    :参数 infix_expression:中缀表达式
    """
    token_list = pred.split()
    # 运算符优先级字典
    pre_dict = {'*': 3, '/': 3, '+': 2, '-': 2, '(': 1}
    # 运算符栈
    operator_stack = []
    # 操作数栈
    operand_stack = []
    for token in token_list:
        # 数字进操作数栈
        if token.isdecimal() or token[1:].isdecimal():
            operand_stack.append(int(token))
        # 左括号进运算符栈
        elif token == '(':
            operator_stack.append(token)
        # 碰到右括号，就要把栈顶的左括号上面的运算符都弹出求值
        elif token == ')':
            top = operator_stack.pop()
            while top != '(':
                # 每弹出一个运算符，就要弹出两个操作数来求值
                # 注意弹出操作数的顺序是反着的，先弹出的数是op2
                op2 = operand_stack.pop()
                op1 = operand_stack.pop()
                # 求出的值要压回操作数栈
                # 这里用到的函数get_value在下面有定义
                operand_stack.append(get_value(top, op1, op2))
                # 弹出下一个栈顶运算符
                top = operator_stack.pop()
        # 碰到运算符，就要把栈顶优先级不低于它的都弹出求值
        elif token in '+-*/':
            while operator_stack and pre_dict[operator_stack[-1]] >= pre_dict[token]:
                top = operator_stack.pop()
                op2 = operand_stack.pop()
                op1 = operand_stack.pop()
                operand_stack.append(get_value(top, op1, op2))
            # 别忘了最后让当前运算符进栈
            operator_stack.append(token)
    # 表达式遍历完成后，栈里剩下的操作符也都要求值
    while operator_stack:
        top = operator_stack.pop()
        op2 = operand_stack.pop()
        op1 = operand_stack.pop()
        operand_stack.append(get_value(top, op1, op2))
    # 最后栈里只剩下一个数字，这个数字就是整个表达式最终的结果
    return 1 if operand_stack[0] == eval(predResult) else 0


def get_value(operator: str, op1: int, op2: int):
    """
    这是四则运算函数
    :参数 operator:运算符
    :参数 op1:左边的操作数
    :参数 op2:右边的操作数
    """
    if operator == '+':
        return op1 + op2
    elif operator == '-':
        return op1 - op2
    elif operator == '*':
        return op1 * op2
    elif operator == '/':
        return op1 / op2


def judgeRotate(img):
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = imgG.shape
    lu = np.sum(imgG[0:int(height / 2), 0:int(width / 2)], dtype=float)
    ld = np.sum(imgG[int(height / 2):, 0:int(width / 2)], dtype=float)
    ru = np.sum(imgG[0:int(height / 2), int(width / 2):], dtype=float)
    rd = np.sum(imgG[int(height / 2):, int(width / 2):], dtype=float)
    if lu + rd - ld - ru > 20000:
        return 'd', 1
    elif ld + ru - lu - rd > 20000:
        return 'u', 1
    return 0


def drawRotatedBox(img, equationImg, orient, pos, color, thickness=4):
    height, width, channel = equationImg.shape
    oriHeight, oriWidth, _ = img.shape
    xmin, ymin, xmax, ymax = pos
    if orient == 'u':
        cv2.line(img, (xmin, max(int(ymin - height / 2), 1)), (xmax, int(ymax - height / 2)), color,
                 thickness=thickness)
        cv2.line(img, (xmin, int(ymin + height / 2)), (xmax, min(int(ymax + height / 2), oriHeight - 1)), color,
                 thickness=thickness)
        cv2.line(img, (xmin, max(int(ymin - height / 2), 1)), (xmin, int(ymin + height / 2)), color,
                 thickness=thickness)
        cv2.line(img, (xmax, int(ymax - height / 2)), (xmax, min(int(ymax + height / 2), oriHeight - 1)), color,
                 thickness=thickness)
    else:
        cv2.line(img, (xmin, int(ymax - height / 2)), (xmax, max(int(ymin - height / 2), 1)), color,
                 thickness=thickness)
        cv2.line(img, (xmin, min(int(ymax + height / 2), oriHeight - 1)), (xmax, int(ymin + height / 2)), color,
                 thickness=thickness)
        cv2.line(img, (xmin, int(ymax - height / 2)), (xmin, min(int(ymax + height / 2), oriHeight - 1)), color,
                 thickness=thickness)
        cv2.line(img, (xmax, max(int(ymin - height / 2), 1)), (xmax, int(ymin + height / 2)), color,
                 thickness=thickness)


def modify(data, yolo, crnnModel, path, modifyImg=False, thickness=4, cvtColor=False, byte=True, video=False):
    """

    :param video:
    :param data: 输入的数据，可以是二进制或array，如果是二进制则要将byte参数置为True
    :param yolo: yolo模型
    :param crnnModel:
    :param path: 照片保存路径
    :param modifyImg: bool，是否矫正图片
    :param thickness: 矩形框宽度
    :param cvtColor: 是否BGR2RGB
    :param byte: 见data参数
    :return:
    """

    imgCount[0], imgCount[1] = 0, 0
    if video and sum(videoCount) != 0:
        videoCount[0], videoCount[1] = 0, 0
    if byte:
        img = np.array(Image.open(BytesIO(data)))
    else:
        img = data
    results = yolo(img).pandas().xyxy[0]
    if type(img) == str:
        img = cv2.imread(img)
    for perEquation in range(len(results)):
        equation = results.iloc[perEquation, :]
        if equation['confidence'] < 0.5:
            continue
        xmin = int(equation['xmin'])
        ymin = int(equation['ymin'])
        xmax = int(equation['xmax'])
        ymax = int(equation['ymax'])
        color = (41, 42, 213)
        equationImg = img[ymin:ymax, xmin:xmax, :]
        result = crnnModel.ocr(equationImg, cls=False, det=False)[0]
        if crnnPaddle(result):
            if video:
                videoCount[0] += 1
            else:
                imgCount[0] += 1
            color = (106, 187, 102)
        else:
            if video:
                videoCount[1] += 1
            else:
                imgCount[1] += 1
        if modifyImg:
            try:
                orient, flag = judgeRotate(equationImg)
                if flag:
                    drawRotatedBox(img, equationImg, orient, (xmin, ymin, xmax, ymax), color, thickness=thickness)
            except Exception as e:
                print(e)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
                pass
        else:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
    if cvtColor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if byte:
        plt.imsave(path, img)
    else:
        return img
