from lib import *
import os

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])
images=glob.glob(os.path.join('./train/time','*.jpg'))
#######MODEL###########
model = models.googlenet(weights='DEFAULT')
model.fc = nn.Linear(in_features=1024, out_features=2)
model.classifier = nn.Linear(in_features=4096, out_features=2)
model.eval()
model.load_state_dict(torch.load('./saved_model/GoogleNet.pth'))
###################
def caculator():
    preds = []
    start_time = time.time()
    for i in images:
        test_img = Image.open(i)
        test_img = test_img.resize((224, 224))
        # print(np.array(test_img).shape)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
        test_img = input_transform(test_img)
        test_img.unsqueeze_(0)
        # test_img = test_img.to(device)
        output = model(test_img)
        _, pred = torch.max(output, 1)
        pred  = pred.tolist()
        preds.append(pred)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    count=0
    for i in preds:
        if(i[0]==1):
            count+=1
    print(count)
    return elapsed_time

time_list = []

for i in range(10):
    time_list.append(caculator())
AVG_time = sum(time_list)/10
print("AVG time: "+ str(AVG_time))