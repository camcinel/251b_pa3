import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


def make_images(model, dataset, palette, index=0, device='cpu'):
    to_image = transforms.ToPILImage()
    palette_tens = torch.tensor(palette).reshape(-1, 3)
    input, truth = dataset[index]
    output = model.forward(torch.unsqueeze(input.to(device=device), 0))
    input_image_rgb = input.permute(1, 2, 0)
    truth_image_rgb = palette_tens[truth.to('cpu')]
    output_image_rgb = palette_tens[torch.argmax(output, dim=1).squeeze().to('cpu')]

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(to_image(input_image_rgb.numpy().astype('uint8')))
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(1, 3, 2)
    plt.imshow(to_image(output_image_rgb.numpy().astype('uint8')))
    plt.axis('off')
    plt.title('Output')

    fig.add_subplot(1, 3, 3)
    plt.imshow(to_image(truth_image_rgb.numpy().astype('uint8')))
    plt.axis('off')
    plt.title('Ground Truth')

    fig.show()


def test_image_transforms(dataset, palette, index=0, device='cpu'):
    to_image = transforms.ToPILImage()
    palette_tens = torch.tensor(palette).reshape(-1, 3)
    input, truth = dataset[index]
    input_image_rgb = input.permute(1, 2, 0)
    truth_image_rgb = palette_tens[truth.to('cpu')]

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(to_image(input_image_rgb.numpy().astype('uint8')))
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(1, 2, 2)
    plt.imshow(to_image(truth_image_rgb.numpy().astype('uint8')))
    plt.axis('off')
    plt.title('Ground Truth')

    fig.show()
