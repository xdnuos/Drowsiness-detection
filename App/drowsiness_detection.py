from lib import *

# Input format of the model.
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

# To load the parameters of the trained model, first we need to initialize same model that we have created into the running application.
# model = DrowsinessCNN()
# model = models.vgg16(weights='VGG16_Weights.DEFAULT')
# model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
model = models.googlenet(weights='GoogLeNet_Weights.DEFAULT')
model.fc = nn.Linear(in_features=1024, out_features=2)
model.classifier = nn.Linear(in_features=4096, out_features=2)
model.eval()

# Loading the parameters of the model - WEIGHTS, BIASES and more.
model.load_state_dict(torch.load('./saved_model/GoogleNet.pth'))
class_index = ["Awake", "Sleep"]

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict_max(self, output): # [0.9, 0.1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.clas_index[max_id]
        return predicted_label
prediccc = Predictor(class_index)

# frame = cv.imread("closed._0.jpg")
# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
# frame = input_transform(frame)
# frame = frame.unsqueeze_(0)

# output = model(frame)
# response = prediccc.predict_max(output)
###########################################
# Capturing the video input from webcam.


# Global Variables
right_eye = None
left_eye = None

# Model output variables.
output_right = None
output_left = None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def define_input(right_eye_input, left_eye_input):

    global right_eye
    global left_eye

    right_eye_image = Image.fromarray(right_eye_input)
    right_eye = right_eye_image.resize((145, 145))

    left_eye_image = Image.fromarray(left_eye_input)
    left_eye = left_eye_image.resize((145, 145))

    right_eye = input_transform(right_eye)
    left_eye = input_transform(left_eye)

    right_eye = right_eye.unsqueeze_(0)
    left_eye = left_eye.unsqueeze_(0)



def beep(response):
    global count
    if(response=="Sleep"):
        count =count +1
    if(response=="Awake"):
        count=0
    if(count >= 3):
        mixer.init()
        mixer.music.load("beep-01a.mp3")
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
count =0
def video():
    cap = cv.VideoCapture(0)
    while True:

        _, frame = cap.read()

        # Get data from VideoCapture(0) - must be in gray format.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect the face inside of the frame
        faces = detector(gray)

        # Iterate faces.
        for face in faces:

            # Apply the landmark to the detected face
            face_lndmrks = predictor(gray, face)

            righteye_x_strt = face_lndmrks.part(37).x-25
            righteye_x_end = face_lndmrks.part(40).x+25
            righteye_y_strt = face_lndmrks.part(20).y-10
            righteye_y_end = face_lndmrks.part(42).y+20

            lefteye_x_strt = face_lndmrks.part(43).x-25
            lefteye_x_end = face_lndmrks.part(46).x+25
            lefteye_y_strt = face_lndmrks.part(25).y-10
            lefteye_y_end = face_lndmrks.part(47).y+20
            cv.rectangle(frame, (righteye_x_strt, righteye_y_strt),
                        (righteye_x_end, righteye_y_end), (0, 255, 0), 2)
            cv.rectangle(frame, (lefteye_x_strt, lefteye_y_strt),
                        (lefteye_x_end, lefteye_y_end), (0, 255, 0), 2)

            right_eye_input = frame[righteye_y_strt:righteye_y_end,
                                    righteye_x_strt:righteye_x_end]
            left_eye_input = frame[lefteye_y_strt:lefteye_y_end,
                                lefteye_x_strt:lefteye_x_end]
            
            current_time = time.localtime().tm_sec
            
            a = threading.Thread(target = define_input, args=(
                right_eye_input, left_eye_input))
            a.start()
            a.join()
            if current_time % 2 == 0:
                output_L = model(left_eye)
                output_R =model(right_eye)
                response_L = prediccc.predict_max(output_L)
                response_R = prediccc.predict_max(output_R)

                if response_L==response_R:
                    cv.putText(frame, response_L, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                    # b = threading.Thread(target = beep, args=(response_L,count))
                    # b.start()
                    # b.join()
                    beep(response_L)
        # Display in the frame
        font= cv.FONT_HERSHEY_SIMPLEX
        text = "Press Q to exit"
        textsize = cv.getTextSize(text, font, 0.5, 1)[0]
        textX = (frame.shape[1] - textsize[0]) / 2
        textX=int(textX)
        textY = 50
        cv.putText(frame, text, (textX, textY), font, 0.5, (255, 255, 0), 1)
        cv.imshow('Frame', frame)

        if cv.waitKey(1) == ord("q"):
            cap.release()
            cv.destroyAllWindows()
            break

def image(image_path):
    img=cv.imread(image_path)

    width = 500
    height = int(img.shape[0] * 500 / img.shape[1])
    dim = (width, height)
    img = cv.resize(img,dim,interpolation = cv.INTER_AREA)
        # Get data from VideoCapture(0) - must be in gray format.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the face inside of the frame
    faces = detector(gray)

    # Iterate faces.
    for face in faces:

        # Apply the landmark to the detected face
        face_lndmrks = predictor(gray, face)

        righteye_x_strt = face_lndmrks.part(37).x-25
        righteye_x_end = face_lndmrks.part(40).x+25
        righteye_y_strt = face_lndmrks.part(20).y-10
        righteye_y_end = face_lndmrks.part(42).y+20

        lefteye_x_strt = face_lndmrks.part(43).x-25
        lefteye_x_end = face_lndmrks.part(46).x+25
        lefteye_y_strt = face_lndmrks.part(25).y-10
        lefteye_y_end = face_lndmrks.part(47).y+20
        cv.rectangle(img, (righteye_x_strt, righteye_y_strt),
                    (righteye_x_end, righteye_y_end), (0, 255, 0), 2)
        cv.rectangle(img, (lefteye_x_strt, lefteye_y_strt),
                    (lefteye_x_end, lefteye_y_end), (0, 255, 0), 2)

        right_eye_input = img[righteye_y_strt:righteye_y_end,
                                righteye_x_strt:righteye_x_end]
        left_eye_input = img[lefteye_y_strt:lefteye_y_end,
                            lefteye_x_strt:lefteye_x_end]
        
        
        a = threading.Thread(target = define_input, args=(
            right_eye_input, left_eye_input))
        a.start()
        a.join()
        output = model(left_eye)
        response = prediccc.predict_max(output)
        cv.putText(img, response, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    # Display in the frame
    cv.imshow('Image', img)