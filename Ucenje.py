#PRVA--KORISTNA STVAR
a = """Gnoj je zlato
     zlato je gnoj!
     
     Ti si sova,
     jaz pa noj."""
     
print(a.split('\n'))
print( '     zlato je gnoj!'.strip())
print( '     '.strip())

#DRUGA--PRIMER PISANJA DATOTEKE IZ SLOVARJA
def pregledno(slovar, ime_datoteke):
    with open(ime_datoteke, 'w', encoding='utf-8') as datoteka:
        for linija in slovar.keys():
            postaje = slovar[linija]
            # Format the line with arrow separators
            vrstica = f"{linija}: {' -> '.join(postaje)}\n"
            datoteka.write(vrstica)

            