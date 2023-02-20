# Author: Adrian Sauter

import copy
import pandas as pd
import random as random
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import json


class Class:
    def __init__(self, name, subjects_and_hours, timeslots):
        """
        create an instance of a class
        #:param subjects_and_hours: list of lists, each sublist contains course name, teacher name and weekly hours
        :param name: name of the class
        :param subjects_and_hours: dictionary, containing course names as keys and teacher name and weekly hours as
        values in a sub-dict
        :param timeslots: weekly time slots available to teach the course
        """
        self.name = name
        self.subjects_and_hours = subjects_and_hours
        self.subjects_and_hours['Break'] = {'hours': 0}
        fill_break_hours(self, len(timeslots) * 5)  # for 5 weekdays
        self.timetable = create_empty_timetable(timeslots)


class Teacher:
    def __init__(self, name, availability, timeslots):
        """
        create an instance of a teacher
        :param name: name of the teacher
        :param availability: list of lists, each sublist contains the availability of the teacher for each
        of the timeslots for each day
        :param timeslots: weekly time slots available to teach the courses
        """
        self.name = name
        self.availability_table = create_availability_table(availability, timeslots)
        self.timetable = create_empty_timetable(timeslots)


def fill_break_hours(class_k, no_timeslots):
    """
    function to fill the remaining timeslots with 'Break'
    :param class_k: the class whose list should be filled
    :param no_timeslots: the total number of timeslots
    :return: -
    """
    num_break_hours = no_timeslots - sum([subdict['hours'] for subdict in class_k.subjects_and_hours.values()])
    if num_break_hours > 0:
        class_k.subjects_and_hours['Break']['hours'] = num_break_hours


def create_availability_table(availability, timeslots):
    """
    :param availability: list of lists, each sublist contains the availability of the teacher for each
        of the timeslots for each day
    :param timeslots: weekly time slots available to teach the courses
    :return: dataframe filled with 0s and 1s, corresponding to the availability of the teacher at that specific timeslot
    """
    availability_table = create_empty_timetable(timeslots)
    for day_idx in range(availability_table.shape[1]):
        for slot_idx, av in zip(range(availability_table.shape[0]), availability[day_idx]):
            availability_table.iat[slot_idx, day_idx] = av
    return availability_table


def create_random_availability(no_timeslots):
    """
    function to create a random availability for a teacher
    :param no_timeslots: number of timeslots per day
    :return: a list with random bits, each bit representing the (un)-availability of a teacher at a certain timeslot
    """
    random_availability = []
    for i in range(5):  # because we have 5 weekdays
        random_availability.append([random.randint(0, 1) for _ in range(no_timeslots)])
    return random_availability


def create_empty_timetable(list_of_timeslots):
    """
    function to create an empty timetable (filled with nan)
    :return: an empty timetable for a class or a teacher (filled with nan)
    """
    return pd.DataFrame(columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                        index=list_of_timeslots)


def reset_timetables(classes, teachers, timeslots):
    """
    function to reset the timetable of all classes and teachers
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :param timeslots: list of timeslots
    :return: -
    """
    for class_k in classes:
        class_k.timetable = create_empty_timetable(timeslots)
    for teacher in teachers:
        teacher.timetable = create_empty_timetable(timeslots)


def fill_timetables(ind, classes, teachers):
    """
    function to fill in the timetables of all classes and teachers with the information from an individual
    :param ind: valid individual
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :return: -
    """
    for class_index, class_k in enumerate(classes):
        for day in range(5):
            for timeslot, code in enumerate(ind[class_index][day]):
                subject = code[0]
                teacher_name = code[1]
                class_k.timetable.iloc[timeslot, day] = (subject, teacher_name)
                for teacher in teachers:
                    if teacher.name == teacher_name:
                        if type(teacher.timetable.iloc[timeslot, day]) == float:
                            teacher.timetable.iloc[timeslot, day] = (subject, class_k.name)
                        elif type(teacher.timetable.iloc[timeslot, day]) == list:
                            teacher.timetable.iloc[timeslot, day] += [(subject, class_k.name)]
                        else:
                            teacher.timetable.iloc[timeslot, day] = [teacher.timetable.iloc[timeslot, day]] + \
                                                                    [(subject, class_k.name)]


def create_example_from_json(filename):
    """
    function to read in data from a .json-file and create Class- and Teacher-objects from the data
    :param filename: name of the .json-file
    :return: list of objects of type 'Class' and list of objects of type 'Teacher'
    """
    f = open(filename)
    dat = json.load(f)
    timeslots = dat['timeslots']
    classes = []
    for class_k in dat['classes']:
        classes.append(Class(class_k['name'], class_k['subjects_and_hours'], timeslots))
    teachers = []
    for teacher in dat['teachers']:
        teachers.append(Teacher(teacher['name'], teacher['availability'], timeslots))
    return classes, teachers


def initialize_random_individual(classes):
    """
    function to initialize a random and valid individual
    :param classes: list of objects of type 'Class'
    :return: randomly initialized, valid individual
    """
    classes_copy = copy.deepcopy(classes)
    sample_class = classes_copy[0]
    num_timeslots = sample_class.timetable.shape[0]
    ind = []
    for class_k in classes_copy:
        class_list = []
        for day in range(5):  # because we have 5 days
            day_list = []
            for timeslot in range(num_timeslots):
                rand_subject = random.choice(list(class_k.subjects_and_hours.keys()))
                while class_k.subjects_and_hours[rand_subject]['hours'] == 0:
                    rand_subject = random.choice(list(class_k.subjects_and_hours.keys()))
                class_k.subjects_and_hours[rand_subject]['hours'] -= 1
                try:
                    corresponding_teacher = class_k.subjects_and_hours[rand_subject]['teacher']
                except KeyError:
                    corresponding_teacher = '-'
                day_list.append((rand_subject, corresponding_teacher))
            class_list.append(day_list)
        ind.append(class_list)
    return ind


def get_fitness_values(population, classes, teachers):
    """
    function to evaluate all individuals from a population
    :param population: list of valid individuals
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :return: list of fitness values, each of which corresponds to an individual in the population
    """
    fitness_vals = []
    sample_class = classes[0]
    timeslots = list(sample_class.timetable.index.values)
    for ind in population:
        fill_timetables(ind, classes, teachers)
        fitness_vals.append(evaluate_individual(classes, teachers))
        reset_timetables(classes, teachers, timeslots)
    return fitness_vals


def evaluate_individual(classes, teachers):
    """
    function to evaluate a single individual
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :return: fitness_score of the individual
    """
    # hard constraints
    fitness_score_double_teacher = check_double_teacher(teachers)
    fitness_score_not_available = teacher_not_available(teachers)
    # soft constraints
    fitness_score_no_classes = zero_classes(classes)
    fitness_score_same_subject_same_day = same_subject_same_day(classes)
    fitness_score_free_timeslots = free_timeslots(classes)
    #print(f'fitness_score_double_teacher: {fitness_score_double_teacher}, '
     #     f'fitness_score_not_available: {fitness_score_not_available}, '
      #    f'fitness_score_no_classes: {fitness_score_no_classes}, '
       #   f'fitness_score_same_subject_same_day: {fitness_score_same_subject_same_day}, '
    #print(f'fitness_score_free_timeslots: {fitness_score_free_timeslots}')
    fitness_score = fitness_score_double_teacher + fitness_score_not_available + fitness_score_no_classes + fitness_score_same_subject_same_day + fitness_score_free_timeslots
                    
    return fitness_score


def check_double_teacher(teachers):
    """
    function to check if a teacher is scheduled for more than one class for a single time slot
    Hard constraint:
    No lecturer can teach two classes at the same time (penalty of 10000).
    :param teachers: list of objects of type 'Teacher'
    :return: a fitness_value, depending on if a teacher teaches in multiple classes at the same time or not
    """
    num_conflicts = 0
    sample_teacher = teachers[0]
    num_days = sample_teacher.timetable.shape[1]
    num_slots = sample_teacher.timetable.shape[0]
    for teacher in teachers:
        for day in range(num_days):
            for slot in range(num_slots):
                if type(teacher.timetable.iloc[slot, day]) == list:
                    num_conflicts += len(teacher.timetable.iloc[slot, day])-1
    return num_conflicts * -10000


def teacher_not_available(teachers):
    """
    function to check if a teacher is unavailable when planned for teaching a course
    Hard constraint:
    No lecturer can teach a class when he/ she is unavailable (penalty of 10000).
    :param teachers: list of objects of type 'Teacher'
    :return: a fitness_value, depending on if a teacher is not available when scheduled for a class
    """
    fitness_score_not_available = 0
    sample_teacher = teachers[0]
    num_days = sample_teacher.timetable.shape[1]
    num_timeslots = sample_teacher.timetable.shape[0]
    for teacher in teachers:
        for day in range(num_days):
            for slot in range(num_timeslots):
                if not type(teacher.timetable.iloc[slot, day]) == float:
                    if teacher.availability_table.iloc[slot, day] == 0:
                        fitness_score_not_available -= 10000
    return fitness_score_not_available


def zero_classes(classes):
    """
    function to determine whether a class doesn't have a single class on a single day
    Soft constraint:
    A class does not have has a single course on a day (penalty of 50 per class that doesn't have a single course
    on a day)
    :param classes: list of objects of type 'Class'
    :return: a fitness_value, depending on how many classes don't have a single subject on a day
    """
    fitness_score_no_classes = 0
    sample_class = classes[0]
    no_days = sample_class.timetable.shape[1]
    no_timeslots = sample_class.timetable.shape[0]
    for day in range(no_days):
        for class_k in classes:
            no_classes_whole_day = True
            for slot in range(no_timeslots):
                subject = class_k.timetable.iloc[slot, day][0]
                if not subject == 'Break':
                    no_classes_whole_day = False
            if no_classes_whole_day:
                fitness_score_no_classes -= 50
    return fitness_score_no_classes


def same_subject_same_day(classes):
    """
    function to determine whether a class has the same subject twice on a day
    Soft constraint:
    A class does shouldn't have the same subject twice on the same day (penalty of 50 per doubling)
    :param classes: list of objects of type 'Class'
    :return: a fitness_value, depending on how many times a class has the same subject twice on a day
    """
    fitness_same_subject_same_day = 0
    sample_class = classes[0]
    no_days = sample_class.timetable.shape[1]
    no_timeslots = sample_class.timetable.shape[0]
    for class_k in classes:
        for day in range(no_days):
            subjects = []
            for slot in range(no_timeslots):
                subject = class_k.timetable.iloc[slot, day][0]
                if not subject == 'Break':
                    if subject in subjects:
                        fitness_same_subject_same_day -= 50
                    else:
                        subjects.append(subject)
    return fitness_same_subject_same_day


def free_timeslots(classes):
    """
    function to determine whether a class has two consecutive free timeslots inbetween courses
    Soft constraint:
    A class does shouldn't have the more than one free timeslot inbetween courses (penalty of 50 per occurrence)
    :param classes: list of objects of type 'Class'
    :return: a fitness_value, depending on how many times a class has more than one free timeslot inbetween classes
    """
    fitness_two_free_timeslots = 0
    sample_class = classes[0]
    no_days = sample_class.timetable.shape[1]
    no_timeslots = sample_class.timetable.shape[0]
    for class_k in classes:
        for day in range(no_days):
            no_break = False
            break_counter = 0
            for slot in range(no_timeslots):
                subject = class_k.timetable.iloc[slot, day][0]
                if subject == 'Break':
                    if not no_break:
                        break_counter += 1
                else:
                    if no_break:        # we have reached another subject, evaluate the number of free timeslots
                        if break_counter >= 2:
                            fitness_two_free_timeslots -= 50 * (break_counter-1)  # -50 for each break more than 1
                        break_counter = 0
                    no_break = True
    return fitness_two_free_timeslots


def get_number_of_conflicts(classes, class_j):
    """
    function to check the number of teacher conflicts (if a teacher is teaching more than one class at a time)
    between classes and class_j
    :param classes: list of objects of type 'Class'
    :param class_j: the class for which the conflicts with classes should be computed
    :return: number of conflicts
    """
    num_conflicts = 0
    num_days = len(class_j)
    num_timeslots = len(class_j[0])
    for day in range(num_days):
        for slot in range(num_timeslots):
            teachers = []
            for class_k in classes:
                teacher = class_k[day][slot][1]  # get the name of the teacher
                if teacher == '-':
                    continue
                else:
                    teachers.append(teacher)
            teacher2 = class_j[day][slot][1]
            if teacher2 == '-':
                continue
            else:
                teachers.append(teacher2)
            teacher_dict = {i: teachers.count(i) for i in teachers}
            for teacher_name in teacher_dict.keys():
                if teacher_dict[teacher_name] > 1:
                    num_conflicts += teacher_dict[teacher_name]-1  # number of conflicts
    return num_conflicts


def crossover(pop, classes, teachers, crossover_prob, smart_crossover, num_parents_for_crossover):
    """
    function to cross over individuals from a population
    :param pop: list of valid individuals
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :param crossover_prob: probability with which a new individual will be created by crossing over two parents
    :param smart_crossover:
    :param num_parents_for_crossover: number of parents to be chosen (only necessary for smart_crossover)
    :return: a new population
    """
    sample_parent = pop[0]
    num_classes = len(sample_parent)
    new_pop = []
    if smart_crossover:
        while len(new_pop) < len(pop):
            child = []
            parents = random.sample(pop, num_parents_for_crossover)
            fitness_vals = get_fitness_values(parents, classes, teachers)
            best_parent = [x for _, x in sorted(zip(fitness_vals, parents), key=lambda x: x[0])][-1]
            child.append(best_parent[0])  # first class from best parent
            for class_k in range(1, num_classes):  # pick rest of the classes according to the least number of conflicts
                num_conflicts = []
                parents_classes = [parent[class_k] for parent in parents]
                for class_j in parents_classes:
                    num_conflicts.append(get_number_of_conflicts(child, class_j))
                child.append([x for _, x in sorted(zip(num_conflicts, parents_classes),
                                                   key=lambda x: x[0])][0])  # choose class with the fewest conflicts
            new_pop.append(child)
    else:
        sample_parent = pop[0]
        num_classes = len(sample_parent)
        new_pop = []
        while len(new_pop) < len(pop):
            parent1, parent2 = random.sample(pop, 2)
            if random.uniform(0, 1) < crossover_prob:
                child = []
                for class_k in range(num_classes):
                    prob = random.randint(0, 1)
                    child.append(parent1[class_k] if prob == 0 else parent2[class_k])
                new_pop.append(child)
            else:      # else: keep the parent with the higher fitness value
                fitness_values = get_fitness_values([parent1, parent2], classes, teachers)
                better_parent = [x for _, x in sorted(zip(fitness_values, [parent1, parent2]), key=lambda x: x[0])][-1]
                new_pop.append(better_parent)
    return new_pop


def mutate(pop, mutation_class_prob, mutation_slot_prob):
    """
    function to randomly perform bitwise-mutation
    :param pop: population
    :param mutation_class_prob: probability with which a class will be mutated
    :param mutation_slot_prob: probability with which a slot will be randomly swapped with another slot within
    the same class
    :return: the mutated population
    """
    for ind_index, ind in enumerate(pop):
        for class_index in range(len(ind)):  # loop over classes
            temp_class_prob = random.uniform(0, 1)
            if temp_class_prob < mutation_class_prob:
                for day in range(len(ind[0])):  # loop over days
                    for timeslot in range(len(ind[0][0])):  # loop over timeslots
                        temp_slot_prob = random.uniform(0, 1)
                        if temp_slot_prob < mutation_slot_prob:
                            code = ind[class_index][day][timeslot]
                            random_day = random.randint(0, len(ind[0])-1)
                            random_timeslot = random.randint(0, len(ind[0][0])-1)
                            exchange_code = ind[class_index][random_day][random_timeslot]
                            ind[class_index][random_day][random_timeslot] = code
                            ind[class_index][day][timeslot] = exchange_code
    return pop


def tournament_selection(population, fitness_vals, num_inds_for_selection=3):
    """
    function to select parents via tournament selection method
    :param population: the previous population
    :param fitness_vals: the fitness values of the individuals of the old population
    :param num_inds_for_selection: number of individuals to be chosen randomly
    :return: the new population
    """
    new_pop = []
    for ind in range(len(population)):
        parents_indices = random.sample(range(len(population)), num_inds_for_selection)
        sorted_fitness_vals = np.argsort([fitness_vals[i] for i in parents_indices])
        parent = population[parents_indices[sorted_fitness_vals[-1]]]  # get individual with the highest fitness value
        new_pop.append(parent)
    return new_pop


def evolutionary_algorithm(classes, teachers,
                           num_individuals, num_iterations,
                           crossover_prob, smart_crossover, num_parents_for_crossover,
                           mutation_class_prob, mutation_slot_prob,
                           keep_elite, num_elite,
                           num_inds_for_tournament):
    """
    function to run the evolutionary algorithm
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :param num_individuals: number of individuals per population/ generation
    :param num_iterations: number of iterations to run
    :param crossover_prob: probability for crossover
    :param smart_crossover: if True, the smart_crossover method will be used
    :param num_parents_for_crossover: number of parents used for crossover (only necessary for smart_crossover)
    :param mutation_class_prob: probability with which a class will be mutated
    :param mutation_slot_prob: probability with which a timeslot will be randomly swapped with another timeslot
    :param keep_elite: if True, the best num_elite individuals are kept in each generation
    :param num_elite: number of individuals to be kept in each generation (with best fitness values)
    :param num_inds_for_tournament: number of individuals to be randomly sampled for each tournament selection
    :return: -
    """
    initial_pop = []
    for _ in range(num_individuals):
        initial_pop.append(initialize_random_individual(classes))
    pop = mutate(initial_pop, mutation_class_prob=1, mutation_slot_prob=1)  # 'shuffle' (in order to make sure that the
                                                                            # 'Break'-subjects are not all stuffed in
                                                                            # the end of the timetables
    fitness_avg_list = []
    best_fitness_val_list = []
    iteration_no = 0
    global_best_fitness_val = float('-inf')
    best_ind = -1
    best_ind_gen = -1
    if num_iterations <= 0:
        raise ValueError('Number of iterations must be at least 1')
    else:
        while iteration_no < num_iterations:
            iteration_no += 1
            before_fitness_vals = get_fitness_values(pop, classes, teachers)
            if not iteration_no == 1:
                print(f'Iteration: {iteration_no}, Average fitness value before mutation and crossover: '
                      f'{np.mean(before_fitness_vals)}, Best fitness value: {sorted(before_fitness_vals)[-1]}')
            fitness_avg_list.append(np.mean(before_fitness_vals))
            if keep_elite:
                pop_copy = copy.deepcopy(pop)  # if the copy isn't made, elite_inds are somehow overwritten
                elite_inds = pop_copy[:num_elite]
                crossover_pop = crossover(pop, classes, teachers,
                                          crossover_prob, smart_crossover, num_parents_for_crossover)
                mutation_pop = mutate(crossover_pop, mutation_class_prob, mutation_slot_prob)
                pop = elite_inds + random.sample(mutation_pop, num_individuals-num_elite)
            else:
                crossover_pop = crossover(pop, classes, teachers,
                                          crossover_prob, smart_crossover, num_parents_for_crossover)
                mutation_pop = mutate(crossover_pop, mutation_class_prob, mutation_slot_prob)
                pop = mutation_pop
            fitness_vals = get_fitness_values(pop, classes, teachers)
            print(f'Iteration: {iteration_no}, Average fitness value after mutation and crossover: '
                  f'{np.mean(fitness_vals)}, Best fitness value: {sorted(fitness_vals)[-1]}')
            print(f'--------------------------------------------------------------------------------------------------')
            pop_best_fitness_val = sorted(fitness_vals)[-1]
            best_fitness_val_list.append(pop_best_fitness_val)
            if pop_best_fitness_val > global_best_fitness_val:
                best_ind_gen = iteration_no
                best_ind = [x for _, x in sorted(zip(fitness_vals, pop), key=lambda y: y[0])][-1]
                global_best_fitness_val = pop_best_fitness_val
            perfect_fitness_indices = [fitness_vals.index(val) for val in fitness_vals if val == 0]
            if len(perfect_fitness_indices) > 0:  # we have at least one individual with a fitness-score of 0
                final_ind = pop[perfect_fitness_indices[0]]
                fill_timetables(final_ind, classes, teachers)
                print_timetables(classes, teachers, iteration_no)
                plot_fitness(fitness_avg_list, iteration_no, avg=True)
                plot_fitness(best_fitness_val_list, iteration_no, avg=False)
                return
            if keep_elite:
                elite_pop = []
                best_fitness_indices = np.argsort(fitness_vals)
                for elite_no in range(1, num_elite+1):
                    elite_pop.append(pop[best_fitness_indices[-elite_no]])
                selection_pop = tournament_selection(pop, fitness_vals, num_inds_for_tournament)
                random_selection = random.sample(selection_pop, num_individuals-num_elite)
                pop = elite_pop + random_selection
            else:
                pop = tournament_selection(pop, fitness_vals, num_inds_for_tournament)
    print(f'The best result was obtained in generation {best_ind_gen} and had the value {global_best_fitness_val}')
    fill_timetables(best_ind, classes, teachers)
    print_timetables(classes, teachers, num_iterations)
    plot_fitness(fitness_avg_list, num_iterations, avg=True)
    plot_fitness(best_fitness_val_list, num_iterations, avg=False)


def create_nice_class_timetable(class_k):
    """
    creates a good-looking timetable for a class
    :param class_k: object of type 'Class'
    :return: -
    """
    num_days = class_k.timetable.shape[1]
    num_timeslots = class_k.timetable.shape[0]
    for day in range(num_days):
        for slot in range(num_timeslots):
            subject = class_k.timetable.iloc[slot, day][0]
            teacher = class_k.timetable.iloc[slot, day][1]
            if teacher == '-':
                class_k.timetable.iloc[slot, day] = 'Break'
            else:
                class_k.timetable.iloc[slot, day] = f'Subject: {subject}\nTeacher: {teacher}'


def create_nice_teacher_timetable(teacher):
    """
    creates a good-looking timetable for a teacher (fond is red if teacher is not available and green if teacher is
    available)
    :param teacher: object of type 'Teacher'
    :return: -
    """
    num_days = teacher.timetable.shape[1]
    num_timeslots = teacher.timetable.shape[0]
    for day in range(num_days):
        for slot in range(num_timeslots):
            try:
                if type(teacher.timetable.iloc[slot, day]) == list:
                    subject = [t[0] for t in teacher.timetable.iloc[slot, day]]
                    class_k = [t[1] for t in teacher.timetable.iloc[slot, day]]
                else:
                    subject = teacher.timetable.iloc[slot, day][0]
                    class_k = teacher.timetable.iloc[slot, day][1]
                if teacher.availability_table.iloc[slot, day] == 0:  # teacher is not available
                    if type(subject) == list:
                        teacher.timetable.iloc[slot, day] = f'\033[31m(Subject: {subject[0]}\033[0m\n' \
                                                                 f'\033[31mClass: {class_k[0]})\033[0m\n'
                        for s, c in zip(subject[1:], class_k[1:]):
                            teacher.timetable.iloc[slot, day] += f'\033[31m(Subject: {s}\033[0m\n' \
                                                                 f'\033[31mClass: {c})\033[0m\n'
                    else:
                        teacher.timetable.iloc[slot, day] = f'\033[31mSubject: {subject}\033[0m\n' \
                                                             f'\033[31mClass: {class_k} \033[0m'
                else:
                    if type(subject) == list:
                        teacher.timetable.iloc[slot, day] = f'\033[32m(Subject: {subject[0]}\033[0m\n' \
                                                            f'\033[32mClass: {class_k[0]})\033[0m\n'
                        for s, c in zip(subject[1:], class_k[1:]):
                            teacher.timetable.iloc[slot, day] += f'\033[32m(Subject: {s}\033[0m\n' \
                                                                 f'\033[32mClass: {c})\033[0m\n'
                    else:
                        teacher.timetable.iloc[slot, day] = f'\033[32mSubject: {subject}\033[0m\n' \
                                                             f'\033[32mClass: {class_k} \033[0m'
            except TypeError:
                if teacher.availability_table.iloc[slot, day] == 0:  # teacher is not available
                    teacher.timetable.iloc[slot, day] = '\033[31m - \033[0m'
                else:
                    teacher.timetable.iloc[slot, day] = '\033[32m - \033[0m'


def print_timetables(classes, teachers, iteration_no):
    """
    print the timetables of all classes and teachers
    :param classes: list of objects of type 'Class'
    :param teachers: list of objects of type 'Teacher'
    :param iteration_no: iteration number at which the timetables are printed
    :return: -
    """
    print(f'Results after {iteration_no} iterations:')
    print('----------------------------------------------------------')
    print('\033[1mClass timetables\033[0m'.center(75))
    for class_k in classes:
        print('\033[1mTimetable of class {}\033[0m'.center(75).format(class_k.name))
        create_nice_class_timetable(class_k)
        print(tabulate(class_k.timetable, headers=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                   'Friday'], tablefmt='fancy_grid'))
        print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('\033[1mTeacher timetables\033[0m'.center(75))
    print('Note that a red-colored text in a teachers timetable means that the teacher is unavailable in that timeslot,'
          ' a green-colored text means that the teacher is available.')
    for teacher in teachers:
        print('\033[1mTimetable of teacher {}\033[0m'.center(75).format(teacher.name))
        create_nice_teacher_timetable(teacher)
        print(tabulate(teacher.timetable, headers=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                   'Friday'], tablefmt='fancy_grid'))
        print('----------------------------------------------------------')
    print('----------------------------------------------------------')


def plot_fitness(fitness_vals, no_iteration, avg=True):
    """
    plot the fitness values
    :param fitness_vals: list of fitness values to be plotted
    :param no_iteration: iteration number at which the fitness values are printed
    :param avg: True if fitness_vals is a list of fitness averages, False if it's a list of the best fitness values
    :return: -
    """
    plt.plot(range(no_iteration), fitness_vals)
    plt.xlabel('Iteration No.')
    if avg:
        plt.ylabel('Average fitness value of population')
        plt.title('Average fitness values for each population')
    else:
        plt.ylabel('Best fitness value of population')
        plt.title('Best fitness values for each population')
    plt.show()


def main():
    classes, teachers = create_example_from_json('easy.json')
    num_individuals = 500
    num_parents_for_crossover = 10
    num_iterations = 500
    crossover_prob = 0.5
    smart_crossover = False
    mutation_class_prob = 0.5
    mutation_slot_prob = 0.33
    keep_elite = False
    num_elite = int(num_individuals/10)
    num_inds_for_tournament = 10
    evolutionary_algorithm(classes=classes, teachers=teachers,
                           num_individuals=num_individuals, num_iterations=num_iterations,
                           crossover_prob=crossover_prob, smart_crossover=smart_crossover,
                           num_parents_for_crossover=num_parents_for_crossover,
                           mutation_class_prob=mutation_class_prob, mutation_slot_prob=mutation_slot_prob,
                           keep_elite=keep_elite, num_elite=num_elite,
                           num_inds_for_tournament=num_inds_for_tournament)


if __name__ == "__main__":
    main()

