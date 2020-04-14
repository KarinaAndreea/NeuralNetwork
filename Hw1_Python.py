
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
            current_helper= []

            free_term = line.split('=')[-1].strip()
            matrix_free_terms.append([int(free_term)])

            equation = line.split('=')[0].strip()
            sign, coefficient, unknown_term = 1, None, None

            for element in equation.split():
                #print(element)
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


# a b c
# d e f
# g h i

def matrix_determinant(matrix):
    a, b, c = matrix[0]
    efhi = [x[1:] for x in matrix[1:]]
    dfgi = [x[::2] for x in matrix[1:]]
    degh = [x[0:2] for x in matrix[1:]]
    determinant = (
            a * det2x2(efhi)
            - b * det2x2(dfgi)
            + c * det2x2(degh)
    )
    return determinant

# determinant of 2x2 matrix
def det2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def calc_transpose(matrix):
   tr_matrix = [[0,0,0],  [0,0,0], [0,0,0]]
   for i in range(len(matrix)):
       for j in range(len(matrix[i])):
           tr_matrix[j][i] = matrix[i][j]
   return tr_matrix

#Find the determinant of each of the 2x2 minor matrices.
def calc_adj(matrix):
    adj_matrix = []
    current_minor = []

    minor_efhi = det2x2([x[1:] for x in matrix[1:]])
    current_minor.append(minor_efhi)
    minor_dfgi =  (-1) * det2x2([x[::2] for x in matrix[1:]])
    current_minor.append(minor_dfgi)
    minor_degh = det2x2([x[0:2] for x in matrix[1:]])
    current_minor.append(minor_degh)
    adj_matrix.append(current_minor)
    current_minor = []


    minor_bchi = (-1) * det2x2([x[1:] for x in matrix[::2]])
    current_minor.append(minor_bchi)
    minor_acgi = det2x2([x[::2] for x in matrix[::2]])
    current_minor.append(minor_acgi)
    minor_abgh = (-1) * det2x2([x[:2] for x in matrix[::2]])
    current_minor.append(minor_abgh)
    adj_matrix.append(current_minor)
    current_minor = []


    minor_bcef = det2x2([x[1:] for x in matrix[:2]])
    current_minor.append(minor_bcef)
    minor_acdf =(-1) * det2x2([x[::2] for x in matrix[:2]])
    current_minor.append(minor_acdf)
    minor_abde = det2x2([x[:2] for x in matrix[:2]])
    current_minor.append(minor_abde)
    adj_matrix.append(current_minor)
    return adj_matrix


def calc_inverse(matrix, det):
    inverse_matrix = [[0,0,0],  [0,0,0], [0,0,0]]
    for row in range(0, len(matrix)):
        for column in range(0, len(matrix[row])):
            inverse_matrix[row][column] = (matrix[row][column] / det)

    return inverse_matrix

def matrix_product(matrix_a, matrix_b):
    matrix_result = [[0],  [0], [0]]
    for i in range(0, len(matrix_a)):
        for j in range(0, len(matrix_b[0])):
            for k in range(0, len(matrix_b)):
                matrix_result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_result

def start(file_name):
    matrix_coef, matrix_free_terms, matrix_unknown_terms, matrix_helper = parse_matrix(file_name)
    # add 0
    for i in range(0, 3):
        for j in matrix_unknown_terms:
            if j not in matrix_helper[i]:
                index = matrix_unknown_terms.index(j)
                matrix_coef[i].insert(index, 0)

    determinant = matrix_determinant(matrix_coef)
    if  determinant == 0:
        print("Determinant is null")
        return

    transpose = calc_transpose(matrix_coef)
    adjoint = calc_adj(transpose)
    inverse = calc_inverse(adjoint, determinant)
    product = matrix_product(inverse, matrix_free_terms)

    print("Solution:")
    for i in range(0, len(matrix_unknown_terms)):
        print("{} = {}".format(matrix_unknown_terms[i], product[i][0]))


start("matrice.txt")
