from PIL import Image
from Convolution_Class import CNN
import torch
import torchvision.transforms as transforms
img = Image.open("number.png")
img = img.convert("L")
img = img.resize((28,28), Image.LANCZOS)
# img.show()
transform = transforms.Compose(
    [transforms.ToTensor()])
tensor = transform(img)
tensor = tensor.unsqueeze_(0)
tensor = tensor.to('cuda')

cnn = CNN().to('cuda')
cnn.load_state_dict(torch.load('my_model_weights.pth'))
cnn.eval()
output = cnn(tensor)

probability = torch.softmax(output[0], dim=1)
print(probability)
predicted_class_index = torch.argmax(probability, dim=1).item()
print(f"The numbers is {predicted_class_index}")
