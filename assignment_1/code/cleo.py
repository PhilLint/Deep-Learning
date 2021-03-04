M = "ICLEOPATRAWILLPROVETOYOUOCAESARTHATMYPEOPLEAREASBRILLIANTASEVERINTHREEMONTHSTIMEIWILLHAVEAMAGNIFICENTPALACEBUILTHEREFORYOUINALEXANDRIA"
M = list(M)
import string
dict = {}

ind = list(range(1, 26))
alph = list(string.ascii_uppercase)

for i in range(len(ind)):
    dict[alph[i]] = ind[i]

K = list("OBELIX")

C = []

def add(l1, l2):
    n3 = None
    n1 = dict[l1]
    n2 = dict[l2]
    n3 = n1 + n2
    if n3 > 26:
        n3 = n3 - 26
    return n3

def get_new_letter(n3):
    new = alph[n3]
    return new

def encrpyt(M, K):
    j = 1
    k = 0

    for i in range(len(M)):
        if j == len(K):
            j = 1
            if k == len(K):
                k = 0
            j +=    k
        print(i)
        print(j)
        n3 = add(M[i], K[j])
        new = get_new_letter(n3)
        C.append(new)
        j += 1

    return C

encrpyt(M, K)







