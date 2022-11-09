from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse, request
from .models import onlineuser
from .models import *
from .ImageProcessing import features_extraction
import matplotlib.pyplot as plt;
import numpy as np
import numpy

global x_train
global y_train
from PIL import ImageTk, Image


#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
                 


# Create your views here.
def homepage(request):
    return render(request, 'index.html')


def userlogindef(request):
    return render(request, 'user.html')


def signupdef(request):
    return render(request, 'signup.html')


def usignupactiondef(request):
    email = request.POST['mail']
    pwd = request.POST['pwd']
    name = request.POST['name']
    ph = request.POST['ph']
    gen = request.POST['gen']
    age = request.POST['age']

    d = onlineuser.objects.filter(email__exact=email).count()
    if d > 0:
        return render(request, 'signup.html', {'msg': "Email Already Registered"})
    else:
        d = onlineuser(name=name, email=email, phone=ph, pwd=pwd, gender=gen, age=age)
        d.save()
        return render(request, 'signup.html', {'msg': "Register Success, You can Login.."})

    return render(request, 'signup.html', {'msg': "Register Success, You can Login.."})


def userloginactiondef(request):
    if request.method == 'POST':
        uid = request.POST['mail']
        pwd = request.POST['pwd']
        d = onlineuser.objects.filter(email__exact=uid).filter(pwd__exact=pwd).count()

        if d > 0:
            d = onlineuser.objects.filter(email__exact=uid)
            name = ""
            for d1 in d:
                name = d1.name

            request.session['useremail'] = uid
            request.session['username'] = name
            return render(request, 'user_home.html', {'data': d[0]})

        else:
            return render(request, 'user.html', {'msg': "Login Details are not matched"})

    else:
        return render(request, 'user.html')


def userhomedef(request):
    if "useremail" in request.session:
        uid = request.session["useremail"]
        d = onlineuser.objects.filter(email__exact=uid)
        return render(request, 'user_home.html', {'data': d[0]})

    else:
        return render(request, 'user.html')


def userlogoutdef(request):
    try:
        del request.session['useremail']
    except:
        pass
    return render(request, 'user.html')


def alogin(request):
    return render(request, 'admin.html')


def adminlogindef(request):
    if request.method == 'POST':
        uid = request.POST['uid']
        pwd = request.POST['pwd']

        if uid == 'admin' and pwd == 'admin':
            request.session['adminid'] = 'admin'
            return render(request, 'admin_home.html')

        else:
            return render(request, 'admin.html', {'msg': "Login Details are not matched"})

    else:
        return render(request, 'admin.html')


def adminhome(request):
    if "adminid" in request.session:

        return render(request, 'admin_home.html')

    else:
        return render(request, 'admin.html')


def alogout(request):
    try:
        del request.session['adminid']
    except:
        pass
    return render(request, 'admin.html')


def training(request):
    if "adminid" in request.session:

        return render(request, 'training.html')

    else:
        return render(request, 'admin.html')


def imagepro(request):
    if "adminid" in request.session:

        x_train, y_train = features_extraction()

        return render(request, 'training.html',
                      {'msg': "Image Processing Completed, Now you can train the algorithms.. "})


    else:
        return render(request, 'admin.html')


def svmtrain(request):
    if "adminid" in request.session:

        from .SVMTraining import build_model
        x_train, y_train = features_extraction()
        build_model(x_train, y_train)

        return render(request, 'training.html', {'msg': "SVM model generated successfully"})


    else:
        return render(request, 'admin.html')


def knntrain(request):
    if "adminid" in request.session:

        from .KNNTraining import build_model
        x_train, y_train = features_extraction()
        build_model(x_train, y_train)

        return render(request, 'training.html', {'msg': "KNN model generated successfully"})


    else:
        return render(request, 'admin.html')


def cnntrain(request):
    if "adminid" in request.session:

        from .Training import train
        train()
        return render(request, 'training.html', {'msg': "CNN model generated successfully"})


    else:
        return render(request, 'admin.html')


def testing(request):
    if "adminid" in request.session:

        return render(request, 'testing.html')

    else:
        return render(request, 'admin.html')


def svmtest(request):
    if "adminid" in request.session:

        from .Svm_Accuracy import calc_svm_accuracy
        acc = calc_svm_accuracy()

        record = performance_sc.objects.filter(algo_name__exact='SVM')
        record.delete()
        record = accuracysc.objects.filter(algo_name__exact='SVM')
        record.delete()
        d = accuracysc.objects.filter(algo_name__exact='SVM').count()
        if d > 0:
            accuracysc.objects.filter(algo_name__exact='SVM').update(accuracy_v=acc[0])
        else:
            print('>>>>>>>>', acc)
            s = accuracysc(algo_name='SVM', accuracy_v=acc[0])
            s.save()
            s = performance_sc(algo_name='SVM', acc_v=acc[0], pre_v=acc[1], rec_v=acc[2], f1_v=acc[3])
            s.save()

        return render(request, 'testing.html',
                      {'msg': "SVM model tested successfully, accuracy is " + str(acc[0]) + " "})


    else:
        return render(request, 'admin.html')


def knntest(request):
    if "adminid" in request.session:

        from .Knn_Accuracy import calc_knn_accuracy
        acc = calc_knn_accuracy()
        record = performance_sc.objects.filter(algo_name__exact='KNN')
        record.delete()
        record = accuracysc.objects.filter(algo_name__exact='KNN')
        record.delete()

        d = accuracysc.objects.filter(algo_name__exact='KNN').count()

        if d > 0:
            accuracysc.objects.filter(algo_name__exact='KNN').update(accuracy_v=acc[0])
        else:
            print('>>>>>>>>', acc)
            s = accuracysc(algo_name='KNN', accuracy_v=acc[0])
            s.save()
            s = performance_sc(algo_name='KNN', acc_v=acc[0], pre_v=acc[1], rec_v=acc[2], f1_v=acc[3])
            s.save()

        return render(request, 'testing.html',
                      {'msg': "KNN model tested successfully, accuracy is " + str(acc[0]) + " "})


    else:
        return render(request, 'admin.html')


def cnntest(request):
    if "adminid" in request.session:
        from .Testing import main
        acc = main()
        record = performance_sc.objects.filter(algo_name__exact='CNN')
        record.delete()
        record = accuracysc.objects.filter(algo_name__exact='CNN')
        record.delete()
        d = accuracysc.objects.filter(algo_name__exact='CNN').count()
        if d > 0:
            accuracysc.objects.filter(algo_name__exact='CNN').update(accuracy_v=acc[0])
        else:
            print('>>>>>>>>', acc)
            s = accuracysc(algo_name='CNN', accuracy_v=acc[0])
            s.save()
            s = performance_sc(algo_name='CNN', acc_v=acc[0], pre_v=acc[1], rec_v=acc[2], f1_v=acc[3])
            s.save()

        return render(request, 'testing.html',
                      {'msg': "CNN model tested successfully, accuracy is " + str(acc[0]) + " "})


    else:
        return render(request, 'admin.html')


def viewacc(request):
    if "adminid" in request.session:
        d = performance_sc.objects.all()

        return render(request, 'viewaccuracy.html', {'data': d})

    else:
        return render(request, 'admin.html')


def viewgraph(request, cat):
    if "adminid" in request.session:
        algorithms = []
        plt.cla()
        plt.clf()

        row = performance_sc.objects.all()
        rlist = []
        for r in row:
            algorithms.append(r.algo_name)
            if cat == 'acc_v':
                rlist.append(r.acc_v)
                plt.title('Accuracy Measure')
            elif cat == 'pre_v':
                rlist.append(r.pre_v)
                plt.title('Precision Measure')
            elif cat == 'rec_v':
                rlist.append(r.rec_v)
                plt.title('Recall Measure')
            elif cat == 'f1_v':
                rlist.append(r.f1_v)
                plt.title('F1-Score Measure')

        height = rlist
        # print(height)
        baars = algorithms
        y_pos = np.arange(len(baars))
        # plt.bar(baars, height, color=['green', 'orange', 'cyan'])
        plt.barh(baars, height, color=['green', 'orange', 'cyan'])
        # plt.plot( baars, height )
        plt.xlabel('')
        plt.ylabel('Algorithms ')
        for index, value in enumerate(height):
            s = str(value)
            s=s[:4]

            plt.text(value, index, s)

        from PIL import Image

        plt.savefig('g1.jpg')
        im = Image.open(r"g1.jpg")
        im.show()

        return redirect('viewacc')




def prediction(request):
    if request.method == 'POST':
        img = request.POST['img']
        img="C:\\Users\\prarthna\\Music\\Django\\SignBoard\\"+img
        image = Image.open(img)
        image = image.resize((30,30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        print(image.shape)
        from keras.models import load_model
        model = load_model('sign_model3.h5')
        print(model)
        pred = model.predict_classes([image])[0]
        print(pred)
        sign = classes[pred+1]

        return render(request, 'result.html', {'data': sign})

 
    else:
        return render(request, 'prediction.html')

