import math
import math
import matplotlib.pyplot as plt

def bisection_method(func, a, b, tolerance, true_value=None, max_iter=100):
  
    if func(a) * func(b) > 0:
        print("Bisection method fails: f(a) and f(b) must have opposite signs.")
        return None

   
    L0 = b - a

    
    theoretical = math.ceil(math.log2(L0 / tolerance))
   # print(f"Initial interval length L0 = {L0}")
    print(f"Theoretical minimum iterations required: {theoretical}\n")

    print(f"{'Iter':<5}{'Lower':<15}{'Midpoint':<15}{'Upper':<15}{'Abs % Error':<20}{'True Error':<20}")
    
    prev_root = None
    abs_errors = []


    root = (a + b) / 2
    for i in range(1, max_iter + 1):
        f_root = func(root)
        abs_percent_error = (L0 / root) * 100 if root != 0 else None
        true_error = abs(true_value - root) if true_value is not None else None

      
        Lk = (b - a)
               
        if prev_root is None:
            es_pct = None
        else:
            es_pct = abs((root - prev_root) / root) * 100

        # True error and absolute error
        if true_value is not None:
            et_pct = ((true_value - root) / true_value) * 100
            ea_pct = abs(et_pct)
        else:
            et_pct = None
            ea_pct = None


       
        abs_percent_error = (Lk / root) * 100 if root != 0 else None
        
      

        
        print(f"{i:<5}{a:<15.10f}{b:<15.10f}{root:<15.10f}"
              f"{Lk:<15.10f}"
              f"{(f'{abs_percent_error:.1-f}%' ):<20}"
              f"{(true_error if true_error is not None else '---')!s:<20}")

        
        if true_value is not None:  
            if true_error is not None and true_error < tolerance:
                break
        else: 
            if Lk < tolerance:
                break

       
        if func(a) * f_root < 0:
            b = root
        else:
            a = root

       
        root = (a + b) / 2

    print("\nFinal Root Approximation:", root)
    print(f"Total Iterations Performed: {i}")
    
    

    return root



if __name__ == "__main__":
   

    
    def f2(x): return  x**3-10*x +5*math.exp(-x/2)-2
   
    bisection_method(f2, a=.1, b=.4, tolerance=.001)
    
    
    
