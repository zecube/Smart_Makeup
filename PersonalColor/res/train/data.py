import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

for filename in glob.glob('spring_bbc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    spring_bbc = f.split(', ')[0:-1]
    spring_bbc = [float(v) for v in spring_bbc]

for filename in glob.glob('summer_bbc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    summer_bbc = f.split(', ')[0:-1]
    summer_bbc = [float(v) for v in summer_bbc]

for filename in glob.glob('fall_bbc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    fall_bbc = f.split(', ')[0:-1]
    fall_bbc = [float(v) for v in fall_bbc]

for filename in glob.glob('winter_bbc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    winter_bbc = f.split(', ')[0:-1]
    winter_bbc = [float(v) for v in winter_bbc]

for filename in glob.glob('spring_sc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    spring_sc = f.split(', ')[0:-1]
    spring_sc = [float(v) for v in spring_sc]

for filename in glob.glob('spring_vc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    spring_vc = f.split(', ')[0:-1]
    spring_vc = [float(v) for v in spring_vc]

for filename in glob.glob('summer_sc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    summer_sc = f.split(', ')[0:-1]
    summer_sc = [float(v) for v in summer_sc]

for filename in glob.glob('summer_vc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    summer_vc = f.split(', ')[0:-1]
    summer_vc = [float(v) for v in summer_vc]

for filename in glob.glob('fall_sc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    fall_sc = f.split(', ')[0:-1]
    fall_sc = [float(v) for v in fall_sc]

for filename in glob.glob('fall_vc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    fall_vc = f.split(', ')[0:-1]
    fall_vc = [float(v) for v in fall_vc]

for filename in glob.glob('winter_sc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    winter_sc = f.split(', ')[0:-1]
    winter_sc = [float(v) for v in winter_sc]

for filename in glob.glob('winter_vc.txt'):
    print(filename)
    f = open(filename, 'r').read()
    winter_vc = f.split(', ')[0:-1]
    winter_vc = [float(v) for v in winter_vc]

warm = spring_bbc + fall_bbc
warm = np.mean(warm)
cool = summer_bbc + winter_bbc
cool = np.mean(cool)

spring_sc = np.mean(spring_sc)
spring_vc = np.mean(spring_vc)
spring_sv = [spring_sc, spring_vc]

summer_sc = np.mean(summer_sc)
summer_vc = np.mean(summer_vc)
summer_sv = [summer_sc, summer_vc]

fall_sc = np.mean(fall_sc)
fall_vc = np.mean(fall_vc)
fall_sv = [fall_sc, fall_vc]

winter_sc = np.mean(winter_sc)
winter_vc = np.mean(winter_vc)
winter_sv = [winter_sc, winter_vc]

print('warm b:', warm)
print('cool b:', cool)
print('spring_sv:', spring_sv)
print('summer_sv:', summer_sv)
print('fall_sv:', fall_sv)
print('winter_sv:', winter_sv)