# experiments/week1_linear_regression.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

# ===== 您不需要理解这些代码 =====
# 生成模拟数据
X = torch.linspace(0, 10, 100).reshape(-1, 1)
Y = 2 * X + 3 + torch.randn(X.size())

# 定义超简单模型
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# ===============================

def main():
    wandb.init(project="ai-learning-journey", name="week1-linear-regression")
    
    # === 只需看这段注释 ===
    print("🎯 本周重点：成功运行这段代码并看到结果")
    print("✅ 不需要理解算法原理（后续会学）")
    print("🔥 目标：体验完整的开发工作流")
    # ====================
    
    # 训练循环（直接运行不需理解）
    epochs = 100
    for epoch in range(epochs):
        output = model(X)
        loss = criterion(output, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({"loss": loss.item()})
    
    # 可视化结果（运行后自动显示图表）
    plt.scatter(X.numpy(), Y.numpy())
    plt.plot(X.numpy(), output.detach().numpy(), 'r')
    plt.savefig("week1_result.png")
    wandb.log({"result": wandb.Image("week1_result.png")})
    wandb.finish()
    print("✨ 任务完成！请查看wandb面板")

if __name__ == "__main__":
    main()