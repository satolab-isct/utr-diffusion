from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torch


def Net_Visible(model):
    x = torch.randn(1, 1, 4, 200, requires_grad=True)  # 示例输入
    time = torch.randint(1, 100, (1,))  # 示例时间输入
    classes = torch.randint(0, 9, (1,))  # 示例类输入

    # model.eval()
    # with torch.no_grad():
    #     output = model(x, time, classes)
    #     print(output.shape)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.render("unet_architecture_neato", format="png", engine='neato')
    # dot.render("unet_architecture_fdp", format="png", engine='fdp')
    # dot.render("unet_architecture_fdp", format="png", engine='sfdp')
    # dot.attr(rankdir='LR')
    # dot.render("unet_architecture--", format="png")
    # dot.render("unet_architecture_svg", format="svg")

    # torch.manual_seed(0)
    writer =SummaryWriter('../../runs/unet_experiment')
    writer.add_graph(model, x)
    writer.close()

