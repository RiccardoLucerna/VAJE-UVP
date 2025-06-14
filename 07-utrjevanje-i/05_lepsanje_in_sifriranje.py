# =============================================================================
# Lepšanje in šifriranje
#
# [Klodovik /papiga/](http://skab612.com/AlanFord/af_likovi.html) in ne
# [Klodvik /frankofonski kralj/](https://sl.wikipedia.org/wiki/Seznam_frankovskih_kraljev)
# bi rad zašifriral svoja besedila, da jih nepoklicane osebe
# ne bodo mogle prebrati.
# =====================================================================@001492=
# 1. podnaloga
# To stori tako, da najprej v besedilu vse male črke spremeni v velike in
# odstrani vse znake, ki niso črke. (Klodovik vsa pomembna besedila piše v
# angleščini. Uporabljali bomo angleško abecedo.) Na primer iz besedila
# `'Attack at dawn!'` dobi besedilo `'ATTACKATDAWN'`. Nato ga zapiše cik-cak
# v treh vrsticah, kot prikazuje primer:
# 
#     A...C...D...
#     .T.A.K.T.A.N
#     ..T...A...W.
# 
# Sestavite funkcijo `cik_cak`, ki sprejme niz in  vrne trojico nizov
# (torej `tuple`) in sicer prvo, drugo in tretjo vrstico v tem zapisu. Primer:
# 
#     >>> cik_cak('Attack at dawn!')
#     ('A...C...D...', '.T.A.K.T.A.N', '..T...A...W.')
# =============================================================================
def cik_cak(niz):
    '''Niz spremenimo v trtvrstični zapis CikCak'''
    perioda = [0, 1, 2, 1] # kako se izmenjujejo nizi, kamor
                           # zapišemo znak: prva, druga, tretja, druga, prva, druga, tretja, druga ...
    niz = niz.upper()
    vrste = ['', '', '']  # vrstice bodo na začetku seznami, da jih lahko spreminjamo!
    stevec = 0 # kateri znak jemljemo 
    for znak in niz:
        if not 'A' <= znak <= 'Z': # če ne gre za znak angleške abecede
            continue
        pos = perioda[stevec % 4] # kam bomo napisali znak
        vrste[pos] += znak
        vrste[(pos+1)%3] += '.'
        vrste[(pos+2)%3] += '.'
        stevec += 1
    return tuple(vrste) # vrniti moramo trojico!
# =====================================================================@001493=
# 2. podnaloga
# Zašifrirano besedilo dobi tako, da najprej prepiše vse znake iz prve
# vrstice, nato vse znake iz druge vrstice in na koncu še vse znake iz
# tretje vrstice. V zgornjem primeru bi tako dobil `'ACDTAKTANTAW'`.
# Sestavite funkcijo `cik_cak_sifra`, ki dobi kot argument niz
# in vrne zašifrirano besedilo. Primer:
# 
#     >>> cik_cak_sifra('Attack at dawn!')
#     'ACDTAKTANTAW'
# =============================================================================
def cik_cak_sifra(s):
    '''Zašifrirajmo besedilo'''
    prva, druga, tretja = cik_cak(s) # najprej zapišemo cik-cak
    sifra = ''
    for znak in prva + druga + tretja: # gremo preko vseh treh vrstic
        if znak != '.': # spustimo pike
            sifra += znak
    return sifra
# =====================================================================@001494=
# 3. podnaloga
# Klodovik se zelo razjezi, ko dobi elektronsko pošto v takšni obliki:
# 
#     Kar sva  si obljubljala    že leta,  si   želiva potrditi tudi   pred prijatelji in   celo
#     žlahto. Vabiva te na
#     
#          poročno slovesnost,        ki bo
#        10.   maja 2016 ob    15.    uri na gradu Otočec.   Prijetno   druženje bomo 
#     nadaljevali v    hotelu   Mons.   Tjaša in  Pavle
# 
# Nepopisno mu gre na živce, da je med besedami po več presledkov. Še
# bolj pa ga nervira, ker so nekatere vrstice precej daljše od drugih.
# Ker je Klodovik vaš dober prijatelj, mu boste pomagali in napisali
# funkcije, s katerimi bo lahko olepšal besedila.
# 
# 
# Najprej napišite funkcijo `razrez`, ki kot argument dobi niz in vrne
# seznam besed v tem nizu. Besede so med seboj ločene z enim ali večimi
# praznimi znaki: `' '` (presledek), `'\t'` (tabulator) in `'\n'` (skok
# v novo vrstico). Pri tej nalogi ločilo obravnavamo kot del besede.
# Primer:
# 
#     >>> razrez('   Kakšen\t pastir, \n\ntakšna  čreda. ')
#     ['Kakšen', 'pastir,', 'takšna', 'čreda.']
# =============================================================================
def razrez(s):
    '''Niz s razreže na podnize, ki jih ločijo "beli" presledki'''
    seznam = []
    beseda = '' # trenutna beseda, ki jo sestavljamo
    for znak in s:
        if znak in ' \n\t':  # znak ni del besede
            if len(beseda) > 0: # če je ta znak zaključil besedo, jo dodamo v seznam
                seznam.append(beseda)
            beseda = '' # in nato bomo začeli sestavljati novo
        else:
            beseda += znak # smo "znotraj" besede
    if len(beseda) > 0: # ne pozabimo na morebitno besedo na koncu!
        seznam.append(beseda)
    return seznam
# =====================================================================@001495=
# 4. podnaloga
# Sedaj, ko že imate funkcijo `razrez`, bo lažje napisati tisto funckijo, ki
# jo Klodovik zares potrebuje. To je
# funkcija `olepsanoBesedilo(s, sir)`, ki kot argumenta dobi niz
# `s` in naravno število `sir`. Funkcija vrne olepšano besedilo, kar
# pomeni naslednje:
# 
# * Funkcija naj odstrani odvečne prazne znake.
# * Vsaka vrstica naj bo kar se le da dolga.
# * Nobena vrstica naj ne vsebuje več kot `sir` znakov (pri čemer znaka
#   `'\n'` na koncu vrstice ne štejemo).
# * Besede znotraj iste vrstice naj bodo ločene s po enim presledkom
#   (ne glede na to, s katerimi in koliko praznimi znaki so ločene v
#   originalnem besedilu).
# 
# Predpostavite, da dolžina nobene besede ni več kot `sir` in da je niz
# `s` neprazen. Primer:
# 
#     >>> s2 = olepsanoBesedilo('  Jasno in   svetlo \t\tna sveti \t\n\nvečer,  dobre\t\t letine je dost, če pa je\t  oblačno in   temno,        žita ne bo.', 20)
#     >>> print(s2)
#     Jasno in svetlo na
#     sveti večer, dobre
#     letine je dost, če
#     pa je oblačno in
#     temno, žita ne bo.
# =============================================================================
def olepsanoBesedilo(s, sir):
    '''olepša besedilo do največje širine sir'''
    besedilo = '' #končno besedilo
    besede = razrez(s) # razrežemo na posamezne besede
    vrstica = ''
    for b in besede:
        if len(vrstica) + len(b) > sir: # če je beseda, ki je na vrsti, predolga
            besedilo += vrstica[:-1] + '\n' # odrežemo presledek, ki smo ga dodali za stikanje
            vrstica = '' # začnemo novo vrstico
        vrstica += b + ' '
    if len(vrstica) > 0: # ne pozabimo na zadnjo vrstico!
        besedilo += vrstica[:-1] + '\n'
    return besedilo[:-1] # na koncu je odvečni prehod v novo vrsto





































































































# ============================================================================@
# fmt: off
"Če vam Python sporoča, da je v tej vrstici sintaktična napaka,"
"se napaka v resnici skriva v zadnjih vrsticah vaše kode."

"Kode od tu naprej NE SPREMINJAJTE!"

# isort: off
import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request
import io
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class TimeoutError(Exception):
    pass


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for x, v in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield

    @staticmethod
    @contextmanager
    def time_limit(timeout_seconds=1):
        from signal import SIGINT, raise_signal
        from threading import Timer

        def interrupt_main():
            raise_signal(SIGINT)

        timer = Timer(timeout_seconds, interrupt_main)
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutError
        finally:
            timer.cancel()


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        part_regex = re.compile(
            r"# =+@(?P<part>\d+)=\s*\n"  # beginning of header
            r"(\s*#( [^\n]*)?\n)+?"  # description
            r"\s*# =+\s*?\n"  # end of header
            r"(?P<solution>.*?)"  # solution
            r"(?=\n\s*# =+@)",  # beginning of next part
            flags=re.DOTALL | re.MULTILINE,
        )
        parts = [
            {"part": int(match.group("part")), "solution": match.group("solution")}
            for match in part_regex.finditer(source)
        ]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]["solution"] = parts[-1]["solution"].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = "{0}.{1}".format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    "part": part["part"],
                    "solution": part["solution"],
                    "valid": part["valid"],
                    "secret": [x for (x, _) in part["secret"]],
                    "feedback": json.dumps(part["feedback"]),
                }
                if "token" in part:
                    submitted_part["token"] = part["token"]
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode("utf-8")
        headers = {"Authorization": token, "content-type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers)
        # This is a workaround because some clients (and not macOS ones!) report
        # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1129)>
        import ssl

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        # When the issue is resolved, the following should be used
        # response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response["attempts"]:
            part["feedback"] = json.loads(part["feedback"])
            updates[part["part"]] = part
        for part in old_parts:
            valid_before = part["valid"]
            part.update(updates.get(part["part"], {}))
            valid_after = part["valid"]
            if valid_before and not valid_after:
                wrong_index = response["wrong_indices"].get(str(part["part"]))
                if wrong_index is not None:
                    hint = part["secret"][wrong_index][1]
                    if hint:
                        part["feedback"].append("Namig: {}".format(hint))

    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoxNDkyLCJ1c2VyIjoxMDc0OH0:1u3HLB:qHsFfHPr1ObanxqzuW8nqI8KTjbYFHD17XKvavq_I3A"
        try:
            Check.equal("""cik_cak('Attack at dawn!')""", ('A...C...D...', '.T.A.K.T.A.N', '..T...A...W.'))
            Check.equal("""cik_cak('We are discovered. Flee at once!')""", ('W...E...C...R...L...T...E', '.E.R.D.S.O.E.E.F.E.A.O.C.', '..A...I...V...D...E...N..'))
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoxNDkzLCJ1c2VyIjoxMDc0OH0:1u3HLB:5-BRq-zvinUznLS9t241I31mHCJWAz-l0amvXs77ZQw"
        try:
            Check.equal("""cik_cak_sifra('Attack at dawn!')""", 'ACDTAKTANTAW')
            Check.equal("""cik_cak_sifra('We are discovered. Flee at once!')""", 'WECRLTEERDSOEEFEAOCAIVDEN')
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoxNDk0LCJ1c2VyIjoxMDc0OH0:1u3HLB:lkCjSMvxO8E2b8-l0nCy2oYDD7Cy_gfaBrqVPTYWuIc"
        try:
            Check.equal("""razrez('Osamelec')""",
                        ['Osamelec'])
            Check.equal("""razrez('Dve besedi')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez('Dve        besedi')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez('    Dve besedi')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez('Dve besedi         ')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez(' Dve  besedi ')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez('\\n\\n\\nDve\\n besedi\t ')""",
                        ['Dve', 'besedi'])
            Check.equal("""razrez('N\\na\\nv\\np\\ni\\nk')""",
                        ['N', 'a', 'v', 'p', 'i', 'k'])
            Check.equal("""razrez('N\\na\\n             v\\np\\ni\\nk')""",
                        ['N', 'a', 'v', 'p', 'i', 'k'])
            Check.equal("""razrez('   Kakšen\\t pastir, \\n\\ntakšna  čreda. ')""",
                        ['Kakšen', 'pastir,', 'takšna', 'čreda.'])
            Check.equal("""razrez('Drevo se po sadu spozna.')""",
                        ['Drevo', 'se', 'po', 'sadu', 'spozna.'])
            Check.equal("""razrez('    Drevo se    po sadu    spozna.    ')""",
                        ['Drevo', 'se', 'po', 'sadu', 'spozna.'])
            Check.equal("""razrez('\\t\\nDrevo \\t\\tse\\t\\tpo\\n\\nsadu spozna.\\n\\n')""",
                        ['Drevo', 'se', 'po', 'sadu', 'spozna.'])
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0IjoxNDk1LCJ1c2VyIjoxMDc0OH0:1u3HLB:CmmLDThSfXqyhrqUehmWRH61ztwNQUhO6cXnz89M8ag"
        try:
            Check.equal("""olepsanoBesedilo('Dve besedi', 6)""",
                        'Dve\nbesedi')
            Check.equal("""olepsanoBesedilo('Dve besedi', 16)""",
                        'Dve besedi')
            Check.equal("""olepsanoBesedilo('\\nDve                besedi   \\t  ', 16)""",
                        'Dve besedi')
            Check.equal("""olepsanoBesedilo('   Dve besedi', 100)""",
                        'Dve besedi')
            Check.equal("""olepsanoBesedilo('N\\na\\nv\\np\\ni\\nk', 3)""",
                        'N a\nv p\ni k')
            Check.equal("""olepsanoBesedilo('N\\na\\nv\\np\\ni\\nk', 6)""",
                        'N a v\np i k')
            Check.equal("""olepsanoBesedilo('N\\na\\nv\\np\\ni\\nk', 5)""",
                        'N a v\np i k')
            Check.equal("""olepsanoBesedilo('N\\na\\nv\\np\\ni', 5)""",
                        'N a v\np i')
            Check.equal("""olepsanoBesedilo('N\\na\\nv\\np\\ni', 6)""",
                        'N a v\np i')
            Check.equal("""olepsanoBesedilo('  Jasno in   svetlo \\t\\tna sveti \\t\\n\\nvečer,  dobre\\t\\t letine je dost, če pa je\\t  oblačno in   temno,        žita ne bo.', 20)""",
                        'Jasno in svetlo na\nsveti večer, dobre\nletine je dost, če\npa je oblačno in\ntemno, žita ne bo.')
            Check.equal("""olepsanoBesedilo('  Jasno in   svetlo \\t\\tna sveti \\t\\n\\nvečer,  dobre\\t\\t letine je dost, če pa je\\t  oblačno in   temno,        žita ne bo.', 7)""",
                        'Jasno\nin\nsvetlo\nna\nsveti\nvečer,\ndobre\nletine\nje\ndost,\nče pa\nje\noblačno\nin\ntemno,\nžita ne\nbo.')
            Check.equal("""olepsanoBesedilo('  Jasno in   svetlo \\t\\tna sveti \\t\\n\\nvečer,  dobre\\t\\t letine je dost, če pa je\\t  oblačno in   temno,        žita ne bo.', 30)""",
                        'Jasno in svetlo na sveti\nvečer, dobre letine je dost,\nče pa je oblačno in temno,\nžita ne bo.')
            Check.equal("""olepsanoBesedilo('  Jasno in   svetlo \\t\\tna sveti \\t\\n\\nvečer,  dobre\\t\\t letine je dost, če pa je\\t  oblačno in   temno,        žita ne bo.', 45)""",
                        'Jasno in svetlo na sveti večer, dobre letine\nje dost, če pa je oblačno in temno, žita ne\nbo.')
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    print("Shranjujem rešitve na strežnik... ", end="")
    try:
        url = "https://www.projekt-tomo.si/api/attempts/submit/"
        token = "Token f301f515dab9db90faf251649f30b362a2143f48"
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        message = (
            "\n"
            "-------------------------------------------------------------------\n"
            "PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE!\n"
            "Preberite napako in poskusite znova ali se posvetujte z asistentom.\n"
            "-------------------------------------------------------------------\n"
        )
        print(message)
        traceback.print_exc()
        print(message)
        sys.exit(1)
    else:
        print("Rešitve so shranjene.")
        update_attempts(Check.parts, response)
        if "update" in response:
            print("Updating file... ", end="")
            backup_filename = backup(filename)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(response["update"])
            print("Previous file has been renamed to {0}.".format(backup_filename))
            print("If the file did not refresh in your editor, close and reopen it.")
    Check.summarize()


if __name__ == "__main__":
    _validate_current_file()
