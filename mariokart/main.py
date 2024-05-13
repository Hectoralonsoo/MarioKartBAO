import csv

import random
import operator
import math
import itertools

import numpy

from functools import partial

class Coche:
    def __init__(self, chasis, ruedas, parapente, personaje):
        self.chasis = chasis
        self.ruedas = ruedas
        self.parapente = parapente
        self.personaje = personaje
        self.peso = chasis.peso + ruedas.peso + parapente.peso + personaje.peso
        self.aceleracion = chasis.aceleracion + ruedas.aceleracion + parapente.aceleracion + personaje.aceleracion
        self.traccion = chasis.traccion + ruedas.traccion + parapente.traccion + personaje.traccion
        self.miniturbo = chasis.traccion + ruedas.traccion + parapente.traccion + personaje.traccion
        self.velTierra = chasis.velTierra + ruedas.velTierra + parapente.velTierra + personaje.velTierra
        self.velAgua = chasis.velAgua + ruedas.velAgua + parapente.velAgua + personaje.velAgua
        self.velAntiGravedad = chasis.velAntiGravedad + ruedas.velAntiGravedad + parapente.velAntiGravedad + personaje.velAntiGravedad
        self.velAire = chasis.velAire + ruedas.velAire + parapente.velAire + personaje.velAire

    def printCoche(self):
        print("Chasis: " + self.chasis.nombre + " Ruedas: " + self.ruedas.nombre + " Parapente: " + self.parapente.nombre + " Personaje: " + self.personaje.nombre)
class Tramo:

    def __init__(self, longitud, terreno, tipo, individuo):
        self.longitud = longitud
        self.terreno = terreno
        self.tipo = tipo
        self.vMax = self.calcularVMax(individuo)

    def calcularVMax(self, individuo):
        if self.tipo == "recta":
            if self.terreno == "tierra":
                return individuo.velTierra
            elif self.terreno == "agua":
                return individuo.velAgua
            elif self.terreno == "aire":
                return individuo.velAire
            else:
                return individuo.velAntiGravedad
        elif self.tipo == "curva cerrada":
            return float(50) + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75
        elif self.tipo == "curva media":
            return 70.0 + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75
        else:
            return 90.0 + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75







class Pieza:
    def __init__(self, nombre, peso , aceleracion , traccion, miniturbo , velTierra , velAgua, velAntiGravedad, velAire):
        self.nombre = nombre
        self.peso = peso
        self.aceleracion = aceleracion
        self.traccion = traccion
        self.miniturbo = miniturbo
        self.velTierra = velTierra
        self.velAgua = velAgua
        self.velAntiGravedad = velAntiGravedad
        self.velAire = velAire

class Chasis(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Rueda(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Parapente(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Personaje(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)

#los numeros en azul son parámetros que habrá que cambiar

def calcularTiempoVuelta(coche, circuito):
    tiempo = 0
    vActual = 0
    i=0
    for tramo in circuito:
        if tramo.tipo=="recta":
            if tramo.terreno=="tierra":
                tiempo += calcularTiempoRecta(int(vActual), coche.velTierra, circuito[i+1].vMax, tramo.longitud, coche.aceleracion, coche.peso, coche.velTierra, coche.traccion)
            elif tramo.terreno=="agua":
                tiempo += calcularTiempoRecta(int(vActual), coche.velAgua, circuito[i + 1].vMax, tramo.longitud, coche.aceleracion, coche.peso, coche.velTierra, coche.traccion)
            elif tramo.terreno=="aire":
                tiempo += calcularTiempoRecta(int(vActual), coche.velAire, circuito[i + 1].vMax, tramo.longitud, coche.aceleracion, coche.peso, coche.velTierra, coche.traccion)
            else:
                tiempo += calcularTiempoRecta(int(vActual), coche.velAntiGravedad, circuito[i + 1].vMax, tramo.longitud, coche.aceleracion, coche.peso, coche.velTierra, coche.traccion)

        else:
            tiempo += calcularTiempoCurva(tramo.longitud, tramo.vMax, Coche.peso, Coche.traccion, Coche.velTierra)
            vActual = tramo.vMax + Coche.miniturbo * 0.9
            # parametrizar impacto del miniturbo

    return tiempo

def calcularTiempoRecta(vInicial, vMax, vCurva, mRecta, aceleracion, peso, velocidad, traccion):
    tiempoRecta = 0
    vCurva = vCurva+float(peso)*0.25+float(traccion)*0.75
    mFrenar = calcularMetros(vCurva, vMax, -aceleracion, peso)
    mVMax = mRecta-mFrenar
    mAcelerar = calcularMetros(vMax, vInicial, aceleracion, peso)
    if(mAcelerar<mVMax):
        mVMax = mRecta-mAcelerar
        tiempoRecta = calcularTiempo(vMax, vInicial, aceleracion, peso) + calcularTiempoVMax(mVMax, velocidad) + calcularTiempo(vCurva, vMax, aceleracion, peso)
    else:
        #en este caso no se alcanza vMax, necesitamos otro método para calcular tiempo acelerando
        tiempoRecta = calcularTiempo(vMax, vInicial, aceleracion, peso) + calcularTiempo(vCurva, vMax, aceleracion, peso)

    return tiempoRecta

def calcularTiempoCurva(mCurva, vCurva, peso, traccion, velocidad):
    if(velocidad<vCurva):
        vCurva = velocidad
    return mCurva * vCurva

def calcularTiempo(vMax, vInicial, aceleracion, peso):
    return (int(vMax)-int(vInicial))/((int(aceleracion)-int(peso))*2)

def calcularTiempoVMax(mVMax, velocidad):
    return mVMax*velocidad*2

def calcularMetros(vFinal, vInicial, aceleracion, peso):
    tiempo =calcularTiempo(vFinal, vInicial, aceleracion, peso)
    return vInicial*tiempo+0.5*aceleracion*10*tiempo*tiempo

gliders = []

with open('gliders.csv', "r") as csvfile:
    csv_reader=csv.reader(csvfile, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = row[1]
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        glider = Parapente(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        gliders.append(glider)

for glider in gliders:
    print(f"Nombre: {glider.nombre}, Peso: {glider.peso}, Aceleracion: {glider.aceleracion}, Traccion: {glider.traccion}, Miniturbo: {glider.miniturbo}, Velocidad en asfalto: {glider.velTierra}, Velocidad  en aire:  {glider.velAire}, Velocidad  en antigravedad:  {glider.velAntiGravedad}, Velocidad en agua: {glider.velAgua}")

tires = []

with open('tires.csv', "r") as csvTires:
    csv_reader = csv.reader(csvTires, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = row[1]
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        tire = Rueda(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        tires.append(tire)

for tire in tires:
    print(f"Nombre: {tire.nombre}, Peso: {tire.peso}, Aceleracion: {tire.aceleracion}, Traccion: {tire.traccion}, Miniturbo: {tire.miniturbo}, Velocidad en asfalto: {tire.velTierra}, Velocidad en agua: {tire.velAgua}, Velocidad en antigravedad: {tire.velAntiGravedad}, Velocidad en aire: {tire.velAire}")

drivers = []

with open('drivers.csv', "r") as csvDrivers:
    csv_reader = csv.reader(csvDrivers, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = row[1]
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        driver = Personaje(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        drivers.append(driver)

for driver in drivers:
    print(f"Nombre: {driver.nombre}, Peso: {driver.peso}, Aceleracion: {driver.aceleracion}, Traccion: {driver.traccion}, Miniturbo: {driver.miniturbo}, Velocidad en asfalto: {driver.velTierra}, Velocidad en agua: {driver.velAgua}, Velocidad en antigravedad: {driver.velAntiGravedad}, Velocidad en aire: {driver.velAire}")

bodies = []

with open('bodies_karts.csv', "r") as csvBodies:
    csv_reader = csv.reader(csvBodies, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = row[1]
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        bodie = Chasis(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        bodies.append(bodie)

for bodie in bodies:
    print(f"Nombre: {bodie.nombre}, Peso: {bodie.peso}, Aceleracion: {bodie.aceleracion}, Traccion: {bodie.traccion}, Miniturbo: {bodie.miniturbo}, Velocidad en asfalto: {bodie.velTierra}, Velocidad en agua: {bodie.velAgua}, Velocidad en antigravedad: {bodie.velAntiGravedad}, Velocidad en aire: {bodie.velAire}")

def generarCoche():
    glider = gliders[random.randint(0, 13)]
    tire = tires[random.randint(0,20)]
    driver = drivers[random.randint(0, 42)]
    body = bodies[random.randint(0, 39)]
    return Coche(body, tire, glider, driver)

def generarPoblacionInicial(size):
    poblacionInicial=[]
    for i in range(size):
        poblacionInicial.append(generarCoche())
    return poblacionInicial

poblacionInicial = generarPoblacionInicial(5)
for coche in poblacionInicial:
    recta1 = Tramo(100, "asfalto", "recta", coche)
    curva1 = Tramo(150, "asfalto", "curva cerrada", coche)
    recta2 = Tramo(100, "asfalto", "recta", coche)
    curva2 = Tramo(150, "asfalto", "curva cerrada", coche)
    ovalo = [recta1, curva1, recta2, curva2]
    tiempo = calcularTiempoVuelta(coche, ovalo)
    coche.printCoche()
    print(" Tiempo total: " + str(tiempo) + "\n")