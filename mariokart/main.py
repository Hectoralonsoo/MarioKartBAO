
import csv
from copy import deepcopy

import numpy as np

from random import Random
from time import time
from inspyred import ec, benchmarks


















class Coche:
    def __init__(self, chasis, ruedas, parapente, personaje):
        self.chasis = chasis
        self.ruedas = ruedas
        self.parapente = parapente
        self.personaje = personaje
        self.peso = chasis.peso + ruedas.peso + parapente.peso + personaje.peso
        self.aceleracion = chasis.aceleracion + ruedas.aceleracion + parapente.aceleracion + personaje.aceleracion
        self.traccion = chasis.traccion + ruedas.traccion + parapente.traccion + personaje.traccion
        self.miniturbo = chasis.miniturbo + ruedas.miniturbo + parapente.miniturbo + personaje.miniturbo
        self.velTierra = chasis.velTierra + ruedas.velTierra + parapente.velTierra + personaje.velTierra
        self.velAgua = chasis.velAgua + ruedas.velAgua + parapente.velAgua + personaje.velAgua
        self.velAntiGravedad = chasis.velAntiGravedad + ruedas.velAntiGravedad + parapente.velAntiGravedad + personaje.velAntiGravedad
        self.velAire = chasis.velAire + ruedas.velAire + parapente.velAire + personaje.velAire

    def printCoche(self):
        print("Chasis: " + self.chasis.nombre + " Ruedas: " + self.ruedas.nombre + " Parapente: " + self.parapente.nombre + " Personaje: " + self.personaje.nombre)

    def printStats(self):
        print("Peso: " + str(self.peso) + "\nAceleración: " + str(self.aceleracion) + "\nTracción: " + str(self.traccion) + "\nMiniturbo: " + str(self.miniturbo) + "\nVelocidad Tierra: " + str(self.velTierra) +
              "\nVelocidad Aire: " + str(self.velAire) + "\nVelocidad Agua: " + str(self.velAgua) + "\nVelocidad Antigravedad: " + str(self.velAntiGravedad))
    def printTraccion(self):
        print("Tracción: " + str(self.traccion))


class Tramo:

    def __init__(self, longitud, terreno, tipo):
        self.longitud = longitud
        self.terreno = terreno
        self.tipo = tipo


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



def calcularTiempoVuelta(coche, circuito):
    velocidadActual = 0
    tiempo = 0
    for tramo in circuito:
        if tramo != circuito[-1]:
            if tramo == "recta":
                tiempo += calcularTiempoRecta(tramo, velocidadActual, calcularVelocidadFinalRecta(coche, tramo), coche.aceleracion)
                velocidadActual = calcularVelocidadFinalRecta(coche, tramo)
            else:
                tiempo += calcularTiempoCurva(tramo, coche)
                velocidadActual = calcularVelocidadFinalCurva(coche, tramo)
        else:
            if tramo == "recta":
                tiempo += calcularTiempoRecta(tramo, velocidadActual, calcularVelocidadFinalRecta(coche, tramo), coche.aceleracion)
            else:
                tiempo += calcularTiempoCurva(tramo, coche)
    return tiempo


def calcularTiempoAcelerando(velocidadInicial, velocidadFinal, aceleracion):
    velocidadMedia = (velocidadInicial + velocidadFinal)/2
    return calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) /velocidadMedia

def calcularTiempoFrenando(velocidadInicial, velocidadFinal, aceleracion):
    velocidadMedia = (velocidadInicial + velocidadFinal)/2
    return calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) /velocidadMedia



def calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion):
    return ((velocidadFinal**2 - velocidadInicial**2) / (2*aceleracion))

def calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion):
    return((velocidadInicial**2-velocidadFinal**2) / (2*aceleracion))

def calcularVelocidadMaximaRecta(tramo, coche):
    if tramo.terreno == "asfalto":
        velocidad = coche.velTierra
    elif tramo.terreno == "agua":
        velocidad = coche.velAgua
    elif tramo.terreno == "aire":
        velocidad = coche.velAire
    else:
        velocidad = coche.velAntiGravedad
    return velocidad * 1.5
def calcularVelocidadFinalRecta(coche, tramo):
    return calcularVelocidadCurva(tramo, coche)
def calcularVelocidadFinalCurva(coche, tramo):
    return 0.5*coche.miniturbo + calcularVelocidadCurva(tramo, coche)
def tiempoDistanciaVelocidadConstante(distancia, velocidad):
    return distancia/velocidad
def calcularTiempoRecta(tramo, velocidadInicial, velocidadFinal, aceleracion):
    return (calcularTiempoAcelerando(velocidadInicial, velocidadFinal, aceleracion) +
            tiempoDistanciaVelocidadConstante(calcularMetrosVelocidadMaximaRecta(tramo)) +
            calcularTiempoFrenando(velocidadInicial,velocidadFinal,aceleracion))
def calcularMetrosVelocidadMaximaRecta(tramo, velocidadInicial, velocidadFinal, aceleracion):
    if tramo.longitud <= ((calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) + calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion))):
        return 0
    else:
        return tramo.longitud - calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) - calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion)


def calcularVelocidadCurva(tramo, coche):
    if tramo.terreno == "asfalto":
        if tramo.tipo == "curva cerrada":
            velocidad = 0.5 * coche.velTierra - coche.peso + coche.traccion * 0.9
        elif tramo.tipo == "curva media":
            velocidad = 0.75 * coche.velTierra - coche.peso + coche.traccion * 0.75
        else:
            velocidad = 0.9 * coche.velTierra - coche.peso + coche.traccion * 0.5

    elif tramo.terreno == "agua":
        if tramo.tipo == "curva cerrada":
            velocidad = 0.5 * coche.velAgua - coche.peso + coche.traccion * 0.9
        elif tramo.tipo == "curva media":
            velocidad = 0.75 * coche.velAgua - coche.peso + coche.traccion * 0.75
        else:
            velocidad = 0.9 * coche.velAgua - coche.peso + coche.traccion * 0.5
    elif tramo.terreno == "aire":
        if tramo.tipo == "curva cerrada":
            velocidad = 0.5 * coche.velAire - coche.peso + coche.traccion * 0.9
        elif tramo.tipo == "curva media":
            velocidad = 0.75 * coche.velAire - coche.peso + coche.traccion * 0.75
        else:
            velocidad = 0.9 * coche.velAire - coche.peso + coche.traccion * 0.5
    else:
        if tramo.tipo == "curva cerrada":
            velocidad = 0.5 * coche.velAntiGravedad - coche.peso + coche.traccion * 0.9
        elif tramo.tipo == "curva media":
            velocidad = 0.75 * coche.velAntiGravedad - coche.peso + coche.traccion * 0.75
        else:
            velocidad = 0.9 * coche.velAntiGravedad - coche.peso + coche.traccion * 0.5

    if velocidad <= 0:
        return 1
    else:
        return velocidad
def calcularTiempoCurva(tramo, coche):
    return tramo.longitud / calcularVelocidadCurva(tramo, coche)


gliders = []

with open('gliders.csv', "r") as csvfile:
    csv_reader=csv.reader(csvfile, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        glider = Parapente(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        gliders.append(glider)


tires = []

with open('tires.csv', "r") as csvTires:
    csv_reader = csv.reader(csvTires, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        tire = Rueda(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        tires.append(tire)



drivers = []

with open('drivers.csv', "r") as csvDrivers:
    csv_reader = csv.reader(csvDrivers, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        driver = Personaje(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        drivers.append(driver)



bodies = []

with open('bodies_karts.csv', "r") as csvBodies:
    csv_reader = csv.reader(csvBodies, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        bodie = Chasis(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        bodies.append(bodie)


def generarCoche(random):
    glider = random.randint(0, 13)
    tire = random.randint(0, 20)
    driver = random.randint(0, 42)
    body = random.randint(0, 39)

    return [body, tire, glider, driver]
def ArrayToCoche(car):
    print(car)
    return Coche(bodies[car[0]],tires[car[1]],gliders[car[2]],drivers[car[3]])






#circuito1

recta1 = Tramo(370, "tierra", "recta")
curva1 = Tramo(150, "agua", "curva cerrada")
recta2 = Tramo(160, "antigravedad", "recta")
curva2 = Tramo(350, "agua", "curva cerrada")
recta3 = Tramo(200, "antigravedad", "recta")
curva3 = Tramo(180, "tierra", "curva media")
recta4 = Tramo(300, "agua", "recta")
curva4 = Tramo(200, "tierra", "curva abierta")

circuito1 = [recta1, curva1, recta2, curva2, recta3, curva3, recta4, curva4]


#circuito 2

recta1 = Tramo(400, "asfalto", "recta")
curva1 = Tramo(200, "agua", "curva cerrada")
curva2 = Tramo(150, "aire", "curva media")
recta2 = Tramo(250, "antigravedad", "recta")
curva3 = Tramo(180, "asfalto", "curva cerrada")
curva4 = Tramo(120, "agua", "curva abierta")
recta3 = Tramo(300, "antigravedad", "recta")
curva5 = Tramo(250, "asfalto", "curva cerrada")
curva6 = Tramo(180, "aire", "curva media")
recta4 = Tramo(280, "asfalto", "recta")
curva7 = Tramo(150, "agua", "curva abierta")
recta5 = Tramo(200, "antigravedad", "recta")

circuito2 = [recta1, curva1, curva2, recta2, curva3, curva4, recta3, curva5, curva6, recta4, curva7, recta5]


#CIRCUITO 3

recta1 = Tramo(600, "asfalto", "recta")
curva1 = Tramo(250, "agua", "curva cerrada")
curva2 = Tramo(300, "aire", "curva media")
recta2 = Tramo(400, "antigravedad", "recta")
curva3 = Tramo(200, "asfalto", "curva cerrada")
curva4 = Tramo(180, "agua", "curva abierta")
recta3 = Tramo(450, "antigravedad", "recta")
curva5 = Tramo(350, "asfalto", "curva cerrada")
curva6 = Tramo(250, "aire", "curva media")
recta4 = Tramo(500, "asfalto", "recta")
curva7 = Tramo(300, "agua", "curva abierta")
recta5 = Tramo(250, "antigravedad", "recta")
curva8 = Tramo(400, "asfalto", "curva cerrada")
curva9 = Tramo(300, "aire", "curva media")
recta6 = Tramo(600, "asfalto", "recta")
curva10 = Tramo(200, "agua", "curva abierta")
recta7 = Tramo(350, "antigravedad", "recta")

circuito3 = [recta1, curva1, curva2, recta2, curva3, curva4, recta3, curva5, curva6, recta4, curva7, recta5, curva8, curva9, recta6, curva10, recta7]



#CIRCUITO 4 CONDICIONES EXTREMAS RECTA AIRE

recta1 = Tramo(1600, "aire", "recta")
curva1 = Tramo(200, "asfalto", "curva cerrada")
recta2 = Tramo(2000, "aire", "recta")
curva2 = Tramo(250, "agua", "curva abierta")
recta3 = Tramo(2500, "aire", "recta")

circuito4 = [recta1, curva1, recta2, curva2, recta3]



#CIRCUITO 5 CONDICIONES EXTREMAS CURVA AGUA Y AIRE

recta1 = Tramo(400, "aire", "recta")
curva1 = Tramo(300, "agua", "curva cerrada")
curva2 = Tramo(250, "aire", "curva abierta")
recta2 = Tramo(500, "aire", "recta")
curva3 = Tramo(300, "agua", "curva cerrada")
recta3 = Tramo(600, "aire", "recta")
curva4 = Tramo(300, "agua", "curva cerrada")
curva5 = Tramo(300, "aire", "curva media")

circuito5 = [recta1, curva1, curva2, recta2, curva3, recta3, curva4, curva5]





class DiscreteBounderV2(object):
    """Defines a basic bounding function for numeric lists of discrete values.

    This callable class acts as a function that bounds a
    numeric list to a set of legitimate values. It does this by
    resolving a given candidate value to the nearest legitimate
    value that can be attained. In the event that a candidate value
    is the same distance to multiple legitimate values, the legitimate
    value appearing earliest in the list will be used.

    For instance, if ``[1, 4, 8, 16]`` was used as the *values* parameter,
    then the candidate ``[6, 10, 13, 3, 4, 0, 1, 12, 2]`` would be
    bounded to ``[4, 8, 16, 4, 4, 1, 1, 8, 1]``.

    Public Attributes:

    - *values* -- the set of attainable values
    - *lower_bound* -- the smallest attainable value
    - *upper_bound* -- the largest attainable value

    """
    def __init__(self, lower_bound,upper_bound):
        self.values = [i for i in range (max(upper_bound))]
        print(self.values)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, candidate, args):
        print("Bounder called with candidate:", candidate)
        bounded_candidate = candidate
        for i in range(len(bounded_candidate)):
            if bounded_candidate[i] < self.lower_bound[i]:
                bounded_candidate[i] = self.lower_bound[i]
            if bounded_candidate[i] > self.upper_bound[i]:
                bounded_candidate[i] = self.upper_bound[i]


        return bounded_candidate

def boundCandidate(candidate):
    bound = [39, 20, 13, 42]
    for i in range(len(candidate)):
        if candidate[i] < 0:
            candidate[i] = 0
        if candidate[i] > bound[i]:
            candidate[i] = bound[i]
    return candidate


class MarioKart(benchmarks.Benchmark):
    """Defines the Mario Kart benchmark problem.

    Public Attributes:

    - *circuito* -- el circuito para el que optimizar el coche


    """
    def __init__(self, circuito):
        benchmarks.Benchmark.__init__(self, 4)
        self.circuito = circuito
        #max_count = [self.capacity // item[0] for item in self.items]
        self.bounder = DiscreteBounderV2([0, 0, 0, 0], [39, 20, 13, 42])
        self.maximize = False

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary algorithm."""
        return generarCoche(random)

    def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            tiempo = calcularTiempoVuelta(ArrayToCoche(candidate), self.circuito)
            coche = Coche(bodies[candidate[0]], tires[candidate[1]], gliders[candidate[2]], drivers[candidate[3]])

            fitness.append(tiempo)
        return fitness


def my_variator(random, candidates, args):
    for candidate in candidates:
        candidate = boundCandidate(candidate)
    return candidates

size = 50


problem = MarioKart(circuito1)

seed = time()  # the current timestamp
prng = Random()
prng.seed(seed)



ga = ec.GA(prng)
ga.selector = ec.selectors.tournament_selection #por defeccto
ga.variator = [ec.variators.n_point_crossover, ec.variators.random_reset_mutation,my_variator] #variators para problema discreto
ga.replacer = ec.replacers.generational_replacement #por defecto
ga.terminator = ec.terminators.generation_termination
ga.observer = ec.observers.stats_observer
final_pop = ga.evolve(generator = problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          pop_size=50,
                          max_generations=60,
                          num_elites=1,
                          num_selected=20,
                          tournament_size=3,
                          crossover_rate=1,
                          sbx_distribution_index=10,
                          mutation_rate=0.9,
                          gaussian_stdev=0.5)

best = max(ga.population)
print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))
mejorCoche = ArrayToCoche(best.candidate)
mejorCoche.printCoche()
mejorCoche.printStats()

class ACOMarioKart:

    def __init__(self, circuito = circuito1, n_ants: int = 10, alpha: float = 1, beta: float = 5, rho: float = 0.8):

        self.circuito = circuito
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone = None
        self.best_solution = None
        self.best_fitness = None

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []


    def optimize(self, max_evaluations: int = 100):
        self._initialize()

        n_evaluations = 0
        iter_fitness = 1e10
        while n_evaluations < max_evaluations:
            trails = []
            for _ in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate(solution)
                n_evaluations += 1
                trails.append((solution, fitness))

                if fitness < self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self._update_pheromone(trails, iter_fitness)
            iter_fitness = self.best_fitness

            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)

        return self.best_solution

    def _initialize(self):
        self.pheromone = []
        self.pheromone.append(np.ones(len(bodies)))
        self.pheromone.append(np.ones(len(tires)))
        self.pheromone.append(np.ones(len(gliders)))
        self.pheromone.append(np.ones(len(drivers)))
        self.best_solution = None
        self.best_fitness = float('inf')

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def _evaluate(self, solution: list[int]) -> float:
        s = ArrayToCoche(solution)
        return calcularTiempoVuelta(s, self.circuito)

    def _construct_solution(self) -> list[int]:
        solution = [-1, -1, -1, -1]
        i = 0
        while i < 4:
            while True:
                candidates = self._get_candidates(i)

                if len(candidates) == 0:
                    break
                elif len(candidates) == 1:
                    solution[i] = candidates[0]
                    break


                pheromones = self.pheromone[i][candidates]**self.alpha
                heuristic = self._heuristic(candidates, i)**self.beta

                total = np.sum(pheromones * heuristic)
                probabilities = (pheromones * heuristic) / total


                solution[i] = np.random.choice(candidates, p=probabilities)
                break
            i = i+1


        return solution

    def _heuristic(self, candidates: list[int], i) -> np.ndarray:
        heuristic = np.zeros(len(candidates))
        piezas = []
        if i == 0:
            piezas = bodies
        elif i == 1:
            piezas = tires
        elif i == 2:
            piezas = gliders
        else: piezas = drivers

        mRecta = 0
        mCurva = 0
        mAsfalto = 0
        mAgua = 0
        mAire = 0
        mAnti = 0
        for tramo in self.circuito:
            if tramo.tipo == "recta":
                mRecta+=tramo.longitud
            else: mCurva+=tramo.longitud

            if tramo.terreno == "asfalto":
                mAsfalto+=tramo.longitud
            elif tramo.terreno == "agua":
                mAgua+=tramo.longitud
            elif tramo.terreno == "aire":
                mAire+=tramo.longitud
            else: mAnti+=tramo.longitud
        for candidate in candidates:
            pieza = piezas[candidate]
            if mAsfalto > mAgua and mAsfalto > mAire and mAsfalto > mAnti:
                heuristic[candidate] = pieza.velTierra
            elif mAgua > mAire and mAgua > mAnti:
                heuristic[candidate] = pieza.velAgua
            elif mAire > mAnti:
                heuristic[candidate] = pieza.velAire
            else:
                heuristic[candidate] = pieza.velAntiGravedad
            if mRecta > mCurva:
                heuristic[candidate] += pieza.aceleracion
            else: heuristic[candidate] += (pieza.traccion + pieza.miniturbo)


        return heuristic

    def _get_candidates(self, nPieza : int) -> np.ndarray:


        if nPieza == 0:
            candidates = [i for i in range(len(bodies))]
        elif nPieza == 1:
            candidates = [i for i in range(len(tires))]
        elif nPieza == 2:
            candidates = [i for i in range(len(gliders))]
        else:
            candidates = [i for i in range(len(drivers))]
        return np.array(candidates)

    def _update_pheromone(self, trails, best_fitness):
        self.pheromone_history.append(self.pheromone.copy())

        evaporation = 1 - self.rho
        self.pheromone[0] *= evaporation
        self.pheromone[1] *= evaporation
        self.pheromone[2] *= evaporation
        self.pheromone[3] *= evaporation
        for solution, fitness in trails:
            delta_fitness = 1.0/(1.0 + (fitness - best_fitness) / best_fitness)
            self.pheromone[0][solution[0]] += delta_fitness
            self.pheromone[1][solution[1]] += delta_fitness
            self.pheromone[2][solution[2]] += delta_fitness
            self.pheromone[3][solution[3]] += delta_fitness


aco = ACOMarioKart(circuito1)
best_solution = aco.optimize()
print("\n Optimización con ACO\n")
print(best_solution)

best_coche = ArrayToCoche(best_solution)
best_coche.printCoche()
best_coche.printStats()
print("Tiempo vuelta: "+str(calcularTiempoVuelta(best_coche, circuito1)))
