from django.shortcuts import render
from django.http import HttpResponse,request
from .models import *
from django.core.mail import send_mail
import mysql
from django.db import connection,IntegrityError,transaction
from mysql.connector import Error
import subprocess
from django.shortcuts import redirect
# Create your views here.
# conne = mysql.connector.connect(user='admin', password='Nikhil2002', host='powersight.cdy8ikaymuro.ap-south-1.rds.amazonaws.com', database='powersight')
conne = mysql.connector.connect(user='root', password='nikhil2002', host='localhost', database='powersight')

def index(request):
    return HttpResponse("Index Dashboard")

def login_page(request):
    return render(request, 'dashboard/loginpage.html')

def register(request):
    if request.method == "POST":
        first_name = request.POST.get('first-name')
        last_name = request.POST.get('last-name')
        phone_number = request.POST.get('phone-number')
        email_address = request.POST.get('email-address')
        password = request.POST.get('password')
        
        try:
            Regis = credent(first_name=first_name,
                    last_name=last_name,
                    phone_number=phone_number,
                    email_address=email_address,
                    password=password)
            Regis.full_clean()
            Regis.save()            
            print("Account saved successfully")

            subject = 'Account Created'
            message = f'Hello {first_name} {last_name},\n\nYour account has been successfully created. You can now login to your Account using your credentials.'
            from_email = 'nchadha_be21@thapar.edu'
            recipient_list = [email_address]
            send_mail(subject, message, from_email, recipient_list)

            success_message = "Account Registered successfully"
            return render(request, 'dashboard/success_not.html', {'message': success_message})
        except IntegrityError:
            error_msg = "Email is already registered"
        except Exception as e:
            error_msg = "An error occurred: {}".format(str(e))
        
        return render(request, 'dashboard/success_not.html', {'message': error_msg})
        
    else:
        return render(request, 'dashboard/loginpage.html')

def dash(request):
    if request.method== "POST":
        Email=request.POST.get("uname")
        password=request.POST.get("psw")

        try:
            conne
            cursor = conne.cursor()
            query = f"SELECT * FROM credentials WHERE email_address = '{Email}' AND password = '{password}'"
            cursor.execute(query)
            user = cursor.fetchone()
            
            if user:
                    first_name_query = f"SELECT first_name FROM credentials WHERE email_address='{Email}'"
                    last_name_query = f"SELECT last_name FROM credentials WHERE email_address='{Email}'"

                    conne
                    cursor = conne.cursor()

                    cursor.execute(first_name_query)
                    first_name_result = cursor.fetchone()
                    first_name = first_name_result[0] if first_name_result else None

                    cursor.execute(last_name_query)
                    last_name_result = cursor.fetchone()
                    last_name = last_name_result[0] if last_name_result else None

                    conne.close()
                    

                    return render(request, 'dashboard/dashdisplay.html', {'first_name': first_name, 'last_name': last_name})
            else:
                conne.close()
                error_message = 'Invalid login credentials. Please try again.'
                return render(request, 'dashboard/success_not.html', {'message': error_message})
        
        except Error as e:
            # Handle database connection or query errors
            error_message = 'An error occurred while accessing the database: {}'.format(str(e))
            return render(request, 'dashboard/success_not.html', {'message': error_message})
        
    else:
        # If request method is GET, show the login page
        return render(request, 'dashboard/loginpage.html')
def resetpass(request):
    if request.method == 'POST':
        email_address = request.POST.get('email')
        prevpass = request.POST.get('prevpass')
        newpass = request.POST.get('newpass')
        conne
        cursor = conne.cursor()
        query = f"UPDATE credentials SET password = '{newpass}' WHERE email_address = '{email_address}' AND password = '{prevpass}'"
        cursor.execute(query)
        if cursor.rowcount > 0:
            # Password was successfully updated in the database
            conne.commit()
            conne.close()
            message = 'Password Reset Successful'
            return render(request, 'dashboard/success_not.html', {'message': message})
        else:
            # Password could not be updated in the database
            conne.rollback()
            conne.close()
            error_message = 'Invalid credentials. Please try again.'
            return render(request, 'dashboard/success_not.html', {'message': error_message})
    else:
        # if request method is GET, show the reset password page
        return render(request, 'dashboard/resetpass.html')
def resetsuccess(request):
    return render(request, 'dashboard/resetsuccess.html')

def streamlit_view(request):
    # Redirect the user to the Streamlit app
    subprocess.Popen(['streamlit', 'run', 'D:\Study\Automated-power-consumption-analysis\power_consumption\dashboard\streamlit\Main.py'])
    
    return render(request,'dashboard/waiting.html')


