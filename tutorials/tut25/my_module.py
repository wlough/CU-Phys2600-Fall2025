"""
Modules can have docstrings, too!
"""


import numpy as my_np

def some_func():
    print("This function is in my_module!")
    
some_data = 44

x = my_np.array([1,2,3])
y = my_np.array([4,5,6])

other_data = x / my_np.exp(x)

print("This line, like the others above, is evaluated on import!")

if __name__ == '__main__':
    main_data = "Won't be imported!"
    
    print("Main sequence running.  (Import won't run this.)")
    print("some_data is: ", some_data)
    print("main_data is: ", main_data)