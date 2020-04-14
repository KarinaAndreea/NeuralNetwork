import numpy as np
def remove_digit(string):
    st_res = ""
    for ch in string:
        if not ch.isdigit():
            st_res +=ch
    return st_res

def remove_let(string):
    st_res = ""
    for ch in string:
        if not ch.isalpha():
            st_res +=ch
    return st_res

def parse_matrix(file):
    matrix_free_terms = []
    matrix_coef =[]
    matrix_unknown_terms = []
    matrix_helper = []

    with open(file, 'r') as fl:
        line = fl.readline()
        while line:
            current_coef = []
            current_helper = []
            free_term = line.split('=')[-1].strip()
            matrix_free_terms.append([int(free_term)])
            equation = line.split('=')[0].strip()
            sign, coefficient, unknown_term = 1, None, None

            for element in equation.split():
                if element == '-':
                    sign = -1
                elif element == '+':
                    continue
                else:
                    coefficient = remove_let(element)
                    if  coefficient == '':
                        coefficient = 1 * sign
                    else:
                        coefficient = int(coefficient) * sign

                    unknown_term = remove_digit(element)
                    current_helper.append(unknown_term)
                    if unknown_term not in matrix_unknown_terms:
                        matrix_unknown_terms.append(unknown_term)

                    current_coef.append(coefficient)
                    sign, coefficient, unknown_term = 1, None, None

            matrix_coef.append(current_coef)
            matrix_helper.append(current_helper)
            line = fl.readline()
    return matrix_coef, matrix_free_terms, matrix_unknown_terms, matrix_helper

def start(file_name):
    matrix_coefficients, matrix_free_terms, matrix_unknown_terms, matrix_helper = parse_matrix(file_name)
    # add 0
    for i in range(0, 3):
        for j in matrix_unknown_terms:
            if j not in matrix_helper[i]:
                index = matrix_unknown_terms.index(j)
                matrix_coefficients[i].insert(index, 0)

    determinant = np.linalg.det(matrix_coefficients)
    if not determinant:
        print("Determinant is null")
        return
    inverse = np.linalg.inv(matrix_coefficients)
    result = np.dot(inverse, matrix_free_terms)
    for i in range(0, len(matrix_unknown_terms)):
        print("{} = {}".format(matrix_unknown_terms[i], result.item(i)))

start("matrice.txt")