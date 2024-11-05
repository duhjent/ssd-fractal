import torch


def main():
    metrics = torch.load("./outs/metrics-smaller.pth")
    print(metrics)


if __name__ == "__main__":
    main()
