import time
from gpiozero import Servo
from pymongo import MongoClient

# MongoDB Configuration
MONGO_URI = "###############################################################"
DATABASE_NAME = "ParkingSystem"
COLLECTION_NAME = "Status"

# GPIO Configuration
SERVO_PIN = 21  # GPIO pin connected to the servo motor

# Servo Configuration
servo = Servo(SERVO_PIN)

# Map degrees to gpiozero's value (-1 to 1)
def degree_to_value(degree):
    return (degree / 180.0) * 2 - 1

# Function to fetch the toll status from MongoDB
def get_toll_status():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        status_doc = collection.find_one({}, {"_id": 0, "status": 1})
        return status_doc.get("status", False)
    except Exception as e:
        print(f"Error fetching status: {e}")
        return False
    finally:
        client.close()

# Function to move the servo to a specific angle
def move_servo_to_angle(angle):
    try:
        print(f"Moving servo to {angle} degrees.")
        servo.value = degree_to_value(angle)
        time.sleep(0.5)  # Wait for the servo to reach the position
    except Exception as e:
        print(f"Error moving servo: {e}")
    finally:
        servo.detach()  # Disable PWM to stop jitter

# Main loop to control the servo
try:
    print("Starting toll control system...")
    current_status = None  # Track the last known status to avoid unnecessary updates
    while True:
        toll_status = get_toll_status()
        if toll_status != current_status:  # Update only if the status changes
            if toll_status:
                move_servo_to_angle(90)  # Open toll
            else:
                move_servo_to_angle(30)  # Close toll
            current_status = toll_status
        time.sleep(2)  # Poll MongoDB every 2 seconds
except KeyboardInterrupt:
    print("Exiting...")
finally:
    servo.detach()  # Ensure the servo is disabled on exit
