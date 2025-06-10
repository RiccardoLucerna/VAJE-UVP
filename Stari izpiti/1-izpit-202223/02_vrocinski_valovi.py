# =============================================================================
# Vročinski valovi
#
# Meteorologi na ARSO so vas prosili za pomoč pri obdelavi meritev. Zanimajo
# jih vročinski valovi v Ljubljani. Poslali so vam dve datoteki: eno s
# povprečnimi temperaturami zraka in eno z temperaturami v tleh na različnih
# globinah. Datoteki vsebujeta dnevne meritve za isto časovno obdobje,
# urejeno po datumih, pri čemer noben datum ni izpuščen.
# =====================================================================@033484=
# 1. podnaloga
# Datoteka s temperaturami zraka vsebuje štiri stolpce: številko merilne
# postaje, ime postaje, datum in temperaturo zraka. Datoteka s temperaturami
# tal vsebuje deset stolpcev: leto, mesec, dan in meritve temperature tal na
# različnih globinah.
# 
# Primer začetka datoteke s temperaturami zraka:
# 
#       stationID,stationName,date,temp
#       _1895,LJUBLJANA - BEZIGRAD,01/01/1992,-3.8
#       _1895,LJUBLJANA - BEZIGRAD,02/01/1992,0.2
#       _1895,LJUBLJANA - BEZIGRAD,03/01/1992,1.7
#       _1895,LJUBLJANA - BEZIGRAD,04/01/1992,
#       _1895,LJUBLJANA - BEZIGRAD,05/01/1992,
#       _1895,LJUBLJANA - BEZIGRAD,06/01/1992,1
#       _1895,LJUBLJANA - BEZIGRAD,07/01/1992,2.8
# 
# Primer začetka datoteke s temperaturami tal:
# 
#        leto,mesec,dan,Ttla2cm,Ttla5cm,Ttla10cm,Ttla20cm,Ttla30cm,Ttla50cm,Ttla100cm
#        1992,1,1,-3.3,,-2.3,-1.7,,1.2,3.9
#        1992,1,2,-2,-2.1,-1.6,-1.3,-0.1,1.1,3.9
#        1992,1,3,-2.4,-2.6,-2.3,-1.6,-0.3,1,
#        1992,1,4,-1.2,-1.5,,-1.2,-0.3,0.9,3.8
#        1992,1,5,-0.9,-1,-0.8,-0.9,-0.2,1,3.8
#        1992,1,6,,,-1,-0.7,-0.1,,3.7
#        1992,1,7,-1.4,-1.5,-1,-0.9,-0.1,1,
# 
# Napišite funkcijo `temperature(vhodna_zrak, vhodna_tla)`, ki sprejme ime
# datoteke s podatki o temperaturah zraka in ime datoteke s podatki o
# temperaturah tal. Funkcija naj vrne seznam trojic s prebranimi podatki.
# Prvi element vsake trojice naj bo datum (niz) oblike dd/mm/yyyy
# (kot je zapisan v prvi datoteki). Drugi element naj bo temperatura zraka,
# tretji pa temperatura tal na globini 50 cm. Temperature naj bodo realna
# števila. Manjkajoče meritve temperatur nadomestite z vrednostjo `-99.9`.
# Seznam naj bo urejen po datumih, torej tako kot si podatki sledijo v datotekah.
# Primer:
# 
#     >>> temperature("zrak.txt", "tla.txt")
#     [('01/01/1992', -3.8, 1.2),
#      ('02/01/1992', 0.2, 1.1),
#      ('03/01/1992', 1.7, 1.0),
#      ('04/01/1992', -99.9, 0.9),
#      ('05/01/1992', -99.9, 1.0),
#      ('06/01/1992', 1.0, -99.9),
#      ('07/01/1992', 2.8, 1.0)]
# =============================================================================

# =====================================================================@033485=
# 2. podnaloga
# Vročinski val v Ljubljani definirajo kot obdobje, ko je izmerjena povprečna
# dnevna temperatura zraka višja od 24 °C vsaj tri dni zapored. Vaša naloga
# je, da določite začetke vročinskih valov. Nalogo lahko rešujete, kot da
# manjkajočih meritev temperatur ni.
# 
# Napišite funkcijo `valovi(podatki)`, ki sprejme tak seznam, kot ga vrne
# funkcija iz prejšnje podnaloge (z datumi in meritvami). Funkcija
# `valovi(nabor)` naj vrne seznam datumov začetkov vročinskih valov
# (to je prvi datum, ko je preseženih 24 °C v posameznem vročinskem valu,
# ki traja vsaj tri zaporedne dni). Datumi naj bodo urejeni naraščajoče.
# Primer:
# 
#      >>> valovi(temperature("zrak.txt", "tla.txt"))
#      ['31/07/1992', '06/08/1992', '18/08/1992', '26/08/1992', '03/08/1993']
# =============================================================================

# =====================================================================@033486=
# 3. podnaloga
# Agrometeorologe zanima še, kako vročinski valovi vplivajo na temperaturo tal,
# ker to vpliva na rast rastlin. Ker jih zanima, kako se običajno spreminja
# temperatura v tleh, vas prosijo, da za prve tri dni vročinkih valov izračunate,
# kakšna je povprečna temperatura na posamezen dan.
# 
# Napišite funkcijo `povprecja(podatki)`, ki sprejme tak seznam, kot pri prejšnji
# podnalogi (z datumi in meritvami). Funkcija naj vrne nabor, ki vsebuje tri
# števila: povprečno temperaturo tal na globini 50 cm na dan začetka vročinskega
# vala, naslednji dan in še na dan za tem. Povprečja naj bodo zaokrožena na eno
# decimalko. Če v vhodnih podatkih ni nobenega vročinskega vala, naj funkcija
# vrne (-99.9, -99.9, -99.9). Primer:
# 
#      >>> povprecja([
#              ('01/08/1992', 15, 12),
#              ('02/08/1992', 20, 10),
#              ('03/08/1992', 25, 8),
#              ('04/08/1992', 26, 10),
#              ('05/08/1992', 25, 11),
#              ('06/08/1992', 20, 10),
#              ('07/08/1992', 19, 10),
#              ('08/08/1992', 26, 12),
#              ('09/08/1992', 26, 11),
#              ('10/08/1992', 26, 11),
#              ('11/08/1992', 22, 10)])
#      (10.0, 10.5, 11.0)
# =============================================================================






































































































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
        ] = "eyJwYXJ0IjozMzQ4NCwidXNlciI6MTA3NDh9:1uP4HN:nO8Bj270ogb2Tl0w9bUITkKNIC47U1kzf2mr4gjFSJQ"
        try:
            input_zrak_short = [
                'stationID,stationName,date,temp',
                '_1895,LJUBLJANA - BEZIGRAD,01/01/1992,-3.8',
                '_1895,LJUBLJANA - BEZIGRAD,02/01/1992,0.2',
                '_1895,LJUBLJANA - BEZIGRAD,03/01/1992,1.7',
                '_1895,LJUBLJANA - BEZIGRAD,04/01/1992,',
                '_1895,LJUBLJANA - BEZIGRAD,05/01/1992,',
                '_1895,LJUBLJANA - BEZIGRAD,06/01/1992,1',
                '_1895,LJUBLJANA - BEZIGRAD,07/01/1992,2.8'
            ]
            
            input_tla_short = [
                'leto,mesec,dan,Ttla2cm,Ttla5cm,Ttla10cm,Ttla20cm,Ttla30cm,Ttla50cm,Ttla100cm',
                '1992,1,1,-3.3,,-2.3,-1.7,,1.2,3.9',
                '1992,1,2,-2,-2.1,-1.6,-1.3,-0.1,1.1,3.9',
                '1992,1,3,-2.4,-2.6,-2.3,-1.6,-0.3,1,',
                '1992,1,4,-1.2,-1.5,,-1.2,-0.3,0.9,3.8',
                '1992,1,5,-0.9,-1,-0.8,-0.9,-0.2,1,3.8',
                '1992,1,6,,,-1,-0.7,-0.1,,3.7',
                '1992,1,7,-1.4,-1.5,-1,-0.9,-0.1,1,'
            ]
            
            data_short = [
                ('01/01/1992',  -3.8,   1.2),
                ('02/01/1992',   0.2,   1.1),
                ('03/01/1992',   1.7,   1.0),
                ('04/01/1992', -99.9,   0.9),
                ('05/01/1992', -99.9,   1.0),
                ('06/01/1992',   1.0, -99.9),
                ('07/01/1992',   2.8,   1.0)
            ]
            
            with Check.in_file('zrak1.txt', input_zrak_short):
                with Check.in_file('tla1.txt', input_tla_short):
                    Check.equal('temperature("zrak1.txt", "tla1.txt")', data_short)
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
        ] = "eyJwYXJ0IjozMzQ4NSwidXNlciI6MTA3NDh9:1uP4HN:Ygamb2S7DoK6fvruHNcuTrbglAIDKi56ctkJXMYeatg"
        try:
            data_supershort = [
                ('01/08/1992', 15, 12),
                ('02/08/1992', 20, 10),
                ('03/08/1992', 25,  8),
                ('04/08/1992', 26, 10),
                ('05/08/1992', 25, 11),
                ('06/08/1992', 20, 10),
                ('07/08/1992', 19, 10),
                ('08/08/1992', 26, 12),
                ('09/08/1992', 26, 11),
                ('10/08/1992', 26, 11),
                ('11/08/1992', 22, 10)
            ]
            
            Check.equal(f'valovi({data_short})', [])
            Check.equal(f'valovi({data_supershort})', ['03/08/1992', '08/08/1992'])
            
            
            data_long = [
                ('01/01/1992', -3.8,  1.2),
                ('02/01/1992',  0.2,  1.1),
                ('03/01/1992',  1.7,  1.0),
                ('04/01/1992',  1.0,  0.9),
                ('05/01/1992',  2.0,  1.0),
                ('06/01/1992',  1.0,  1.0),
                ('07/01/1992',  2.8,  1.0),
                ('08/01/1992',  6.7,  1.0),
                ('09/01/1992',  7.2,  1.2),
                ('10/01/1992',  6.9,  1.1),
                ('11/01/1992',  5.5,  1.0),
                ('12/01/1992',  2.0,  1.1),
                ('13/01/1992',  1.6,  1.4),
                ('14/01/1992',  2.4,  1.4),
                ('15/01/1992',  1.3,  1.7),
                ('16/01/1992', -1.9,  1.7),
                ('17/01/1992', -0.6,  1.7),
                ('18/01/1992', -2.3,  1.7),
                ('19/01/1992', -0.8,  1.7),
                ('20/01/1992',  0.4,  1.7),
                ('21/01/1992', -4.6,  1.6),
                ('22/01/1992', -3.9,  1.6),
                ('23/01/1992', -2.2,  1.6),
                ('24/01/1992',  0.0,  1.6),
                ('25/01/1992',  0.6,  1.6),
                ('26/01/1992', -1.7,  1.6),
                ('27/01/1992', -1.0,  1.6),
                ('28/01/1992', -0.9,  1.6),
                ('29/01/1992', -0.7,  1.7),
                ('30/01/1992',  0.0,  1.6),
                ('31/01/1992', -0.6,  1.7),
                ('01/02/1992', -0.1,  1.6),
                ('02/02/1992', -0.4,  1.6),
                ('03/02/1992',  1.2,  1.6),
                ('04/02/1992',  1.9,  1.6),
                ('05/02/1992',  3.4,  1.6),
                ('06/02/1992',  3.7,  1.7),
                ('07/02/1992',  3.7,  1.6),
                ('08/02/1992',  2.0,  1.6),
                ('09/02/1992',  2.9,  1.6),
                ('10/02/1992',  3.5,  1.6),
                ('11/02/1992',  4.6,  1.6),
                ('12/02/1992',  6.7,  1.7),
                ('13/02/1992',  6.4,  2.1),
                ('14/02/1992',  7.3,  2.6),
                ('15/02/1992',  6.2,  3.2),
                ('16/02/1992',  6.1,  3.5),
                ('17/02/1992',  2.5,  3.5),
                ('18/02/1992', -0.5,  3.4),
                ('19/02/1992', -2.2,  3.0),
                ('20/02/1992', -1.8,  2.4),
                ('21/02/1992',  2.1,  2.2),
                ('22/02/1992',  2.6,  2.1),
                ('23/02/1992',  3.5,  2.2),
                ('24/02/1992',  4.8,  2.5),
                ('25/02/1992',  4.9,  2.7),
                ('26/02/1992',  6.2,  3.0),
                ('27/02/1992',  6.6,  3.1),
                ('28/02/1992',  5.5,  3.3),
                ('29/02/1992',  6.2,  3.5),
                ('01/03/1992',  7.7,  3.8),
                ('02/03/1992',  7.7,  3.9),
                ('03/03/1992',  8.3,  4.2),
                ('04/03/1992',  9.2,  4.6),
                ('05/03/1992',  8.0,  5.0),
                ('06/03/1992', 10.2,  5.3),
                ('07/03/1992',  8.2,  5.6),
                ('08/03/1992',  4.6,  5.8),
                ('09/03/1992',  4.3,  5.3),
                ('10/03/1992',  4.2,  5.3),
                ('11/03/1992',  5.3,  5.0),
                ('12/03/1992',  6.8,  5.2),
                ('13/03/1992',  8.4,  5.5),
                ('14/03/1992',  7.2,  5.6),
                ('15/03/1992',  5.4,  5.7),
                ('16/03/1992',  5.2,  5.8),
                ('17/03/1992',  4.9,  5.9),
                ('18/03/1992',  4.8,  5.7),
                ('19/03/1992',  6.3,  5.7),
                ('20/03/1992',  7.8,  6.1),
                ('21/03/1992', 12.0,  6.6),
                ('22/03/1992',  5.1,  7.2),
                ('23/03/1992',  6.3,  7.1),
                ('24/03/1992',  8.4,  6.8),
                ('25/03/1992',  7.4,  7.1),
                ('26/03/1992',  0.9,  7.0),
                ('27/03/1992',  3.7,  5.2),
                ('28/03/1992',  2.0,  4.9),
                ('29/03/1992',  4.4,  4.8),
                ('30/03/1992',  3.9,  5.5),
                ('31/03/1992',  6.5,  5.5),
                ('01/04/1992',  7.9,  5.9),
                ('02/04/1992',  7.2,  6.6),
                ('03/04/1992', 10.1,  6.9),
                ('04/04/1992', 12.7,  7.7),
                ('05/04/1992',  9.4,  8.2),
                ('06/04/1992',  6.5,  8.3),
                ('07/04/1992',  8.1,  7.9),
                ('08/04/1992',  9.1,  8.2),
                ('09/04/1992',  9.3,  8.4),
                ('10/04/1992',  9.0,  8.1),
                ('11/04/1992',  9.8,  8.2),
                ('12/04/1992', 10.4,  8.4),
                ('13/04/1992', 12.3,  8.7),
                ('14/04/1992', 10.6,  9.3),
                ('15/04/1992', 11.2,  9.6),
                ('16/04/1992',  6.9,  9.5),
                ('17/04/1992',  5.9,  9.0),
                ('18/04/1992',  8.2,  8.6),
                ('19/04/1992', 13.1,  8.8),
                ('20/04/1992', 10.6,  9.8),
                ('21/04/1992',  9.5,  9.9),
                ('22/04/1992', 11.4, 10.2),
                ('23/04/1992', 11.4, 10.4),
                ('24/04/1992', 13.9, 10.3),
                ('25/04/1992', 15.8, 11.1),
                ('26/04/1992', 17.5, 11.9),
                ('27/04/1992', 17.5, 13.0),
                ('28/04/1992', 17.9, 13.7),
                ('29/04/1992', 11.6, 13.8),
                ('30/04/1992',  7.7, 12.9),
                ('01/05/1992', 12.8, 11.9),
                ('02/05/1992', 14.4, 12.3),
                ('03/05/1992', 17.6, 12.9),
                ('04/05/1992', 15.7, 13.3),
                ('05/05/1992', 15.2, 13.7),
                ('06/05/1992', 17.0, 13.8),
                ('07/05/1992', 16.1, 14.3),
                ('08/05/1992', 17.3, 14.8),
                ('09/05/1992', 18.1, 15.1),
                ('10/05/1992', 18.2, 15.5),
                ('11/05/1992', 14.4, 15.9),
                ('12/05/1992', 14.3, 15.4),
                ('13/05/1992', 16.6, 15.4),
                ('14/05/1992', 18.2, 16.0),
                ('15/05/1992', 18.2, 16.4),
                ('16/05/1992', 19.5, 17.0),
                ('17/05/1992', 19.0, 17.4),
                ('18/05/1992', 15.2, 17.4),
                ('19/05/1992', 11.9, 16.6),
                ('20/05/1992', 11.7, 15.8),
                ('21/05/1992', 13.8, 14.8),
                ('22/05/1992', 14.2, 15.1),
                ('23/05/1992', 13.8, 15.1),
                ('24/05/1992', 17.3, 15.0),
                ('25/05/1992', 17.3, 15.5),
                ('26/05/1992', 16.8, 16.3),
                ('27/05/1992', 17.8, 16.8),
                ('28/05/1992', 16.7, 17.3),
                ('29/05/1992', 17.0, 17.3),
                ('30/05/1992', 16.7, 17.5),
                ('31/05/1992', 17.8, 17.2),
                ('01/06/1992', 18.1, 17.3),
                ('02/06/1992', 21.8, 17.3),
                ('03/06/1992', 19.6, 18.2),
                ('04/06/1992', 18.0, 18.1),
                ('05/06/1992', 18.1, 18.5),
                ('06/06/1992', 15.6, 18.1),
                ('07/06/1992', 15.1, 17.6),
                ('08/06/1992', 14.9, 17.3),
                ('09/06/1992', 15.4, 17.1),
                ('10/06/1992', 15.6, 17.1),
                ('11/06/1992', 15.8, 17.2),
                ('12/06/1992', 14.0, 17.1),
                ('13/06/1992', 15.6, 16.7),
                ('14/06/1992', 19.2, 17.0),
                ('15/06/1992', 20.2, 17.5),
                ('16/06/1992', 20.8, 18.3),
                ('17/06/1992', 20.1, 18.7),
                ('18/06/1992', 20.4, 18.9),
                ('19/06/1992', 19.7, 19.3),
                ('20/06/1992', 19.8, 19.0),
                ('21/06/1992', 20.3, 19.0),
                ('22/06/1992', 21.0, 19.4),
                ('23/06/1992', 20.0, 19.7),
                ('24/06/1992', 20.7, 19.6),
                ('25/06/1992', 18.3, 19.9),
                ('26/06/1992', 18.9, 19.2),
                ('27/06/1992', 18.6, 19.2),
                ('28/06/1992', 17.5, 19.5),
                ('29/06/1992', 20.0, 19.6),
                ('30/06/1992', 21.6, 19.9),
                ('01/07/1992', 20.9, 20.5),
                ('02/07/1992', 20.9, 20.9),
                ('03/07/1992', 22.4, 21.2),
                ('04/07/1992', 22.4, 21.5),
                ('05/07/1992', 16.5, 21.5),
                ('06/07/1992', 17.9, 20.1),
                ('07/07/1992', 18.5, 19.8),
                ('08/07/1992', 19.0, 19.5),
                ('09/07/1992', 18.3, 19.8),
                ('10/07/1992', 18.3, 19.2),
                ('11/07/1992', 18.2, 19.1),
                ('12/07/1992', 17.4, 18.9),
                ('13/07/1992', 20.2, 18.9),
                ('14/07/1992', 20.7, 19.4),
                ('15/07/1992', 22.5, 20.1),
                ('16/07/1992', 19.7, 20.7),
                ('17/07/1992', 20.8, 20.7),
                ('18/07/1992', 22.9, 20.6),
                ('19/07/1992', 22.6, 21.0),
                ('20/07/1992', 22.1, 21.3),
                ('21/07/1992', 22.0, 21.7),
                ('22/07/1992', 23.4, 22.2),
                ('23/07/1992', 23.0, 22.7),
                ('24/07/1992', 23.3, 22.7),
                ('25/07/1992', 22.7, 22.4),
                ('26/07/1992', 23.7, 22.2),
                ('27/07/1992', 25.6, 22.8),
                ('28/07/1992', 22.9, 23.2),
                ('29/07/1992', 20.9, 22.4),
                ('30/07/1992', 22.8, 22.6),
                ('31/07/1992', 26.0, 22.8),
                ('01/08/1992', 26.6, 23.5),
                ('02/08/1992', 25.1, 24.0),
                ('03/08/1992', 26.2, 24.1),
                ('04/08/1992', 23.5, 24.3),
                ('05/08/1992', 23.1, 24.1),
                ('06/08/1992', 24.7, 23.9),
                ('07/08/1992', 25.6, 24.1),
                ('08/08/1992', 25.5, 24.3),
                ('09/08/1992', 26.5, 24.5),
                ('10/08/1992', 22.9, 24.6),
                ('11/08/1992', 21.1, 23.9),
                ('12/08/1992', 21.8, 23.7),
                ('13/08/1992', 22.3, 23.4),
                ('14/08/1992', 23.1, 23.6),
                ('15/08/1992', 18.9, 23.4),
                ('16/08/1992', 21.0, 22.5),
                ('17/08/1992', 23.6, 22.6),
                ('18/08/1992', 24.8, 22.9),
                ('19/08/1992', 26.1, 23.3),
                ('20/08/1992', 26.2, 23.7),
                ('21/08/1992', 26.8, 23.9),
                ('22/08/1992', 20.5, 24.0),
                ('23/08/1992', 21.1, 22.9),
                ('24/08/1992', 22.8, 22.3),
                ('25/08/1992', 23.7, 22.3),
                ('26/08/1992', 24.9, 22.5),
                ('27/08/1992', 25.4, 22.9),
                ('28/08/1992', 25.7, 23.3),
                ('29/08/1992', 24.7, 23.4),
                ('30/08/1992', 19.4, 23.4),
                ('31/08/1992', 20.2, 22.0),
                ('01/09/1992', 16.1, 21.3),
                ('02/09/1992', 15.1, 20.2),
                ('03/09/1992', 17.5, 19.7),
                ('04/09/1992', 10.6, 19.6),
                ('05/09/1992', 12.4, 18.2),
                ('06/09/1992', 12.5, 17.4),
                ('07/09/1992', 13.9, 17.3),
                ('08/09/1992', 17.5, 17.3),
                ('09/09/1992', 16.8, 17.6),
                ('10/09/1992', 17.8, 18.1),
                ('11/09/1992', 19.8, 18.4),
                ('12/09/1992', 19.1, 18.7),
                ('13/09/1992', 18.7, 19.0),
                ('14/09/1992', 18.9, 19.2),
                ('15/09/1992', 17.9, 19.3),
                ('16/09/1992', 18.2, 19.0),
                ('17/09/1992', 17.8, 19.0),
                ('18/09/1992', 16.2, 19.0),
                ('19/09/1992', 14.8, 18.6),
                ('20/09/1992', 14.5, 18.3),
                ('21/09/1992', 14.8, 18.1),
                ('22/09/1992', 17.0, 17.9),
                ('23/09/1992', 17.5, 17.9),
                ('24/09/1992', 15.5, 17.9),
                ('25/09/1992', 15.0, 17.6),
                ('26/09/1992', 15.8, 17.3),
                ('27/09/1992', 17.8, 17.3),
                ('28/09/1992', 16.9, 17.5),
                ('29/09/1992', 16.1, 17.5),
                ('30/09/1992', 16.1, 17.4),
                ('01/10/1992', 13.4, 17.5),
                ('02/10/1992', 13.6, 17.1),
                ('03/10/1992', 11.1, 16.8),
                ('04/10/1992', 12.8, 16.2),
                ('05/10/1992', 14.3, 15.7),
                ('06/10/1992', 14.6, 15.7),
                ('07/10/1992', 15.2, 15.5),
                ('08/10/1992', 13.4, 16.0),
                ('09/10/1992', 13.3, 15.9),
                ('10/10/1992', 12.3, 15.7),
                ('11/10/1992', 10.9, 15.4),
                ('12/10/1992',  8.9, 14.9),
                ('13/10/1992',  4.7, 14.3),
                ('14/10/1992',  3.9, 13.2),
                ('15/10/1992',  7.0, 12.7),
                ('16/10/1992',  9.4, 12.3),
                ('17/10/1992', 12.4, 12.2),
                ('18/10/1992',  4.4, 12.4),
                ('19/10/1992',  3.7, 11.5),
                ('20/10/1992',  6.5, 11.0),
                ('21/10/1992',  8.9, 10.9),
                ('22/10/1992',  6.6, 10.9),
                ('23/10/1992',  6.9, 10.8),
                ('24/10/1992',  7.4, 10.7),
                ('25/10/1992',  9.1, 10.6),
                ('26/10/1992', 10.7, 10.6),
                ('27/10/1992',  7.7, 10.8),
                ('28/10/1992', 11.1, 10.7),
                ('29/10/1992',  9.5, 11.2),
                ('30/10/1992',  5.4, 10.9),
                ('31/10/1992',  7.8, 10.2),
                ('01/11/1992',  7.4, 10.1),
                ('02/11/1992',  8.2, 10.2),
                ('03/11/1992', 11.5, 10.3),
                ('04/11/1992',  8.0, 10.6),
                ('05/11/1992',  8.8, 10.5),
                ('06/11/1992',  7.2, 10.5),
                ('07/11/1992',  7.9, 10.4),
                ('08/11/1992',  8.7, 10.3),
                ('09/11/1992',  7.1, 10.1),
                ('10/11/1992',  8.0,  9.8),
                ('11/11/1992',  9.4,  9.9),
                ('12/11/1992',  5.6, 10.1),
                ('13/11/1992',  3.2,  9.8),
                ('14/11/1992',  2.3,  9.1),
                ('15/11/1992',  4.2,  8.6),
                ('16/11/1992', 11.6,  8.6),
                ('17/11/1992',  7.8,  9.2),
                ('18/11/1992',  2.7,  9.3),
                ('19/11/1992',  1.2,  8.7),
                ('20/11/1992',  1.1,  8.1),
                ('21/11/1992',  0.9,  7.8),
                ('22/11/1992',  6.5,  7.6),
                ('23/11/1992',  6.0,  7.5),
                ('24/11/1992',  3.8,  7.5),
                ('25/11/1992', 11.0,  7.6),
                ('26/11/1992', 10.8,  8.1),
                ('27/11/1992',  6.3,  8.3),
                ('28/11/1992',  7.9,  8.2),
                ('29/11/1992',  6.1,  8.1),
                ('30/11/1992',  5.3,  8.2),
                ('01/12/1992',  5.7,  8.1),
                ('02/12/1992',  8.9,  8.2),
                ('03/12/1992', 11.4,  8.3),
                ('04/12/1992', 11.5,  8.8),
                ('05/12/1992', 10.3,  8.9),
                ('06/12/1992',  3.5,  8.9),
                ('07/12/1992',  1.1,  8.1),
                ('08/12/1992',  2.4,  7.2),
                ('09/12/1992',  1.2,  6.6),
                ('10/12/1992',  2.5,  6.2),
                ('11/12/1992',  2.4,  6.1),
                ('12/12/1992',  2.0,  6.2),
                ('13/12/1992', -0.4,  5.8),
                ('14/12/1992', -0.8,  5.7),
                ('15/12/1992', -1.1,  5.4),
                ('16/12/1992', -1.4,  5.3),
                ('17/12/1992', -0.2,  5.0),
                ('18/12/1992',  0.0,  5.1),
                ('19/12/1992', -1.1,  4.9),
                ('20/12/1992', -2.0,  4.8),
                ('21/12/1992',  0.1,  4.5),
                ('22/12/1992',  1.2,  4.6),
                ('23/12/1992',  0.8,  4.7),
                ('24/12/1992', -3.1,  4.7),
                ('25/12/1992', -5.5,  4.4),
                ('26/12/1992', -5.5,  4.1),
                ('27/12/1992', -6.7,  3.6),
                ('28/12/1992', -2.2,  3.3),
                ('29/12/1992', -6.4,  2.9),
                ('30/12/1992', -6.9,  2.8),
                ('31/12/1992', -6.5,  2.5),
                ('01/01/1993', -4.9,  2.0),
                ('02/01/1993', -7.6,  1.7),
                ('03/01/1993', -6.1,  1.5),
                ('04/01/1993', -5.2,  1.3),
                ('05/01/1993', -6.8,  1.3),
                ('06/01/1993', -6.4,  1.1),
                ('07/01/1993', -2.8,  1.0),
                ('08/01/1993',  1.0,  1.0),
                ('09/01/1993',  0.7,  1.0),
                ('10/01/1993', -1.4,  1.0),
                ('11/01/1993',  6.1,  1.2),
                ('12/01/1993',  8.4,  1.2),
                ('13/01/1993',  6.1,  1.3),
                ('14/01/1993',  6.9,  1.3),
                ('15/01/1993',  2.6,  1.3),
                ('16/01/1993',  2.2,  1.5),
                ('17/01/1993',  4.7,  1.7),
                ('18/01/1993',  2.0,  1.9),
                ('19/01/1993', -0.3,  2.1),
                ('20/01/1993', -1.0,  2.1),
                ('21/01/1993',  0.2,  2.2),
                ('22/01/1993',  4.0,  2.2),
                ('23/01/1993',  5.3,  2.5),
                ('24/01/1993',  7.7,  2.7),
                ('25/01/1993',  6.3,  3.4),
                ('26/01/1993',  1.3,  3.4),
                ('27/01/1993',  3.8,  3.1),
                ('28/01/1993',  1.7,  2.8),
                ('29/01/1993',  5.1,  2.7),
                ('30/01/1993', -1.1,  3.0),
                ('31/01/1993', -3.7,  2.9),
                ('01/02/1993', -3.0,  2.6),
                ('02/02/1993', -2.3,  2.3),
                ('03/02/1993', -0.4,  2.1),
                ('04/02/1993',  0.1,  1.9),
                ('05/02/1993',  4.9,  1.8),
                ('06/02/1993',  5.9,  1.8),
                ('07/02/1993',  4.3,  1.7),
                ('08/02/1993',  3.9,  1.8),
                ('09/02/1993',  1.3,  1.7),
                ('10/02/1993',  0.9,  1.8),
                ('11/02/1993',  0.9,  1.8),
                ('12/02/1993',  0.8,  1.8),
                ('13/02/1993', -0.5,  1.8),
                ('14/02/1993', -1.7,  1.7),
                ('15/02/1993',  1.7,  1.7),
                ('16/02/1993', -0.4,  1.7),
                ('17/02/1993', -3.7,  1.7),
                ('18/02/1993',  0.1,  1.7),
                ('19/02/1993',  1.8,  1.6),
                ('20/02/1993',  4.1,  1.6),
                ('21/02/1993',  1.3,  1.6),
                ('22/02/1993', -0.5,  1.6),
                ('23/02/1993', -1.6,  1.6),
                ('24/02/1993',  1.0,  1.6),
                ('25/02/1993',  0.9,  1.5),
                ('26/02/1993',  0.6,  1.5),
                ('27/02/1993',  1.1,  1.5),
                ('28/02/1993',  5.3,  1.5),
                ('01/03/1993',  3.1,  1.5),
                ('02/03/1993',  0.9,  1.5),
                ('03/03/1993',  0.7,  1.7),
                ('04/03/1993', -0.8,  1.8),
                ('05/03/1993', -2.5,  1.9),
                ('06/03/1993', -0.3,  1.9),
                ('07/03/1993',  0.6,  1.9),
                ('08/03/1993',  2.8,  2.0),
                ('09/03/1993',  3.5,  2.2),
                ('10/03/1993',  4.7,  2.6),
                ('11/03/1993',  6.4,  2.8),
                ('12/03/1993',  6.8,  3.0),
                ('13/03/1993',  7.1,  3.5),
                ('14/03/1993',  7.3,  4.2),
                ('15/03/1993',  8.4,  4.1),
                ('16/03/1993',  9.4,  4.6),
                ('17/03/1993', 10.2,  4.9),
                ('18/03/1993', 14.1,  5.3),
                ('19/03/1993', 11.3,  5.9),
                ('20/03/1993', 13.3,  6.3),
                ('21/03/1993', 12.8,  6.6),
                ('22/03/1993', 11.8,  7.3),
                ('23/03/1993', 12.8,  7.6),
                ('24/03/1993',  5.9,  8.0),
                ('25/03/1993',  3.5,  7.7),
                ('26/03/1993',  3.7,  7.0),
                ('27/03/1993',  4.0,  6.5),
                ('28/03/1993',  4.5,  6.3),
                ('29/03/1993',  4.9,  6.1),
                ('30/03/1993',  4.6,  6.3),
                ('31/03/1993',  5.7,  6.1),
                ('01/04/1993',  8.6,  6.3),
                ('02/04/1993',  7.7,  6.7),
                ('03/04/1993',  4.5,  6.9),
                ('04/04/1993',  7.8,  6.9),
                ('05/04/1993',  9.4,  7.3),
                ('06/04/1993',  9.6,  7.6),
                ('07/04/1993', 11.2,  8.4),
                ('08/04/1993',  7.3,  8.4),
                ('09/04/1993',  9.0,  8.3),
                ('10/04/1993',  9.1,  8.6),
                ('11/04/1993',  6.5,  9.0),
                ('12/04/1993',  4.8,  8.6),
                ('13/04/1993',  7.8,  8.2),
                ('14/04/1993', 11.9,  8.5),
                ('15/04/1993',  8.8,  8.6),
                ('16/04/1993', 10.7,  9.1),
                ('17/04/1993',  8.7,  9.4),
                ('18/04/1993', 11.0,  9.4),
                ('19/04/1993', 12.7,  9.5),
                ('20/04/1993', 15.7,  9.7),
                ('21/04/1993', 15.2, 10.6),
                ('22/04/1993', 16.3, 11.3),
                ('23/04/1993', 15.8, 11.9),
                ('24/04/1993', 15.7, 12.2),
                ('25/04/1993', 15.9, 12.7),
                ('26/04/1993', 17.5, 12.9),
                ('27/04/1993', 14.9, 13.4),
                ('28/04/1993', 15.1, 13.2),
                ('29/04/1993', 14.7, 13.3),
                ('30/04/1993', 13.0, 13.3),
                ('01/05/1993', 13.2, 13.1),
                ('02/05/1993', 13.7, 13.0),
                ('03/05/1993', 15.5, 13.2),
                ('04/05/1993', 14.8, 13.6),
                ('05/05/1993', 12.7, 13.8),
                ('06/05/1993', 13.9, 13.6),
                ('07/05/1993', 14.1, 13.5),
                ('08/05/1993', 14.5, 13.4),
                ('09/05/1993', 14.4, 13.6),
                ('10/05/1993', 15.1, 13.8),
                ('11/05/1993', 17.4, 14.1),
                ('12/05/1993', 15.5, 14.5),
                ('13/05/1993', 15.1, 14.7),
                ('14/05/1993', 15.5, 15.0),
                ('15/05/1993', 14.2, 15.1),
                ('16/05/1993', 15.1, 14.9),
                ('17/05/1993', 16.8, 15.2),
                ('18/05/1993', 18.7, 15.7),
                ('19/05/1993', 19.3, 16.5),
                ('20/05/1993', 19.7, 17.0),
                ('21/05/1993', 19.8, 17.5),
                ('22/05/1993', 17.8, 17.7),
                ('23/05/1993', 17.4, 17.9),
                ('24/05/1993', 19.9, 18.0),
                ('25/05/1993', 21.5, 18.3),
                ('26/05/1993', 22.2, 18.6),
                ('27/05/1993', 21.8, 19.5),
                ('28/05/1993', 21.8, 19.3),
                ('29/05/1993', 17.3, 19.2),
                ('30/05/1993', 20.6, 18.3),
                ('31/05/1993', 19.0, 18.4),
                ('01/06/1993', 21.1, 18.3),
                ('02/06/1993', 21.3, 18.4),
                ('03/06/1993', 14.3, 19.1),
                ('04/06/1993', 17.2, 18.1),
                ('05/06/1993', 20.5, 17.8),
                ('06/06/1993', 22.0, 18.4),
                ('07/06/1993', 19.9, 18.9),
                ('08/06/1993', 23.6, 19.9),
                ('09/06/1993', 23.4, 19.5),
                ('10/06/1993', 23.6, 20.2),
                ('11/06/1993', 22.4, 20.5),
                ('12/06/1993', 18.0, 20.7),
                ('13/06/1993', 15.8, 20.3),
                ('14/06/1993', 17.5, 19.5),
                ('15/06/1993', 18.2, 19.3),
                ('16/06/1993', 18.9, 19.5),
                ('17/06/1993', 16.8, 19.8),
                ('18/06/1993', 18.6, 19.2),
                ('19/06/1993', 21.5, 19.3),
                ('20/06/1993', 22.7, 20.3),
                ('21/06/1993', 18.2, 20.6),
                ('22/06/1993', 21.9, 20.1),
                ('23/06/1993', 19.3, 20.3),
                ('24/06/1993', 15.8, 20.0),
                ('25/06/1993', 16.8, 19.3),
                ('26/06/1993', 17.0, 19.2),
                ('27/06/1993', 16.6, 19.0),
                ('28/06/1993', 16.2, 18.8),
                ('29/06/1993', 16.7, 18.6),
                ('30/06/1993', 19.9, 18.8),
                ('01/07/1993', 20.9, 19.4),
                ('02/07/1993', 22.1, 20.0),
                ('03/07/1993', 23.5, 20.6),
                ('04/07/1993', 24.7, 21.2),
                ('05/07/1993', 25.3, 21.7),
                ('06/07/1993', 16.0, 22.0),
                ('07/07/1993', 17.0, 20.7),
                ('08/07/1993', 19.4, 20.2),
                ('09/07/1993', 20.7, 20.2),
                ('10/07/1993', 20.2, 20.5),
                ('11/07/1993', 17.6, 20.4),
                ('12/07/1993', 13.5, 20.2),
                ('13/07/1993', 15.6, 19.5),
                ('14/07/1993', 16.6, 19.4),
                ('15/07/1993', 19.4, 19.4),
                ('16/07/1993', 21.9, 19.6),
                ('17/07/1993', 24.0, 20.3),
                ('18/07/1993', 24.6, 21.1),
                ('19/07/1993', 23.7, 21.6),
                ('20/07/1993', 21.6, 21.9),
                ('21/07/1993', 17.7, 21.8),
                ('22/07/1993', 16.0, 21.1),
                ('23/07/1993', 18.0, 20.4),
                ('24/07/1993', 20.8, 20.2),
                ('25/07/1993', 23.0, 20.7),
                ('26/07/1993', 13.8, 20.9),
                ('27/07/1993', 17.8, 19.7),
                ('28/07/1993', 22.1, 19.8),
                ('29/07/1993', 24.0, 20.2),
                ('30/07/1993', 25.0, 21.0),
                ('31/07/1993', 25.8, 21.6),
                ('01/08/1993', 21.2, 22.0),
                ('02/08/1993', 23.0, 21.9),
                ('03/08/1993', 24.9, 22.0),
                ('04/08/1993', 26.0, 22.4),
                ('05/08/1993', 26.3, 22.7),
                ('06/08/1993', 23.0, 23.0),
                ('07/08/1993', 22.1, 22.8),
                ('08/08/1993', 19.9, 22.9),
                ('09/08/1993', 16.9, 21.6),
                ('10/08/1993', 20.4, 20.8),
                ('11/08/1993', 20.3, 20.8),
                ('12/08/1993', 21.0, 20.9),
                ('13/08/1993', 22.6, 21.2),
                ('14/08/1993', 23.1, 21.5),
                ('15/08/1993', 24.3, 21.8),
                ('16/08/1993', 24.3, 22.1),
                ('17/08/1993', 22.5, 22.4),
                ('18/08/1993', 20.3, 22.4),
                ('19/08/1993', 22.0, 21.9),
                ('20/08/1993', 23.2, 22.1),
                ('21/08/1993', 24.6, 22.1),
                ('22/08/1993', 25.3, 22.5),
                ('23/08/1993', 26.0, 22.7),
                ('24/08/1993', 22.8, 22.9),
                ('25/08/1993', 15.2, 22.7),
                ('26/08/1993', 13.9, 21.1),
                ('27/08/1993', 14.4, 20.0),
                ('28/08/1993', 14.4, 19.4),
                ('29/08/1993', 14.6, 18.8),
                ('30/08/1993', 13.5, 18.4),
                ('31/08/1993', 14.8, 17.7),
                ('01/09/1993', 14.4, 18.4),
                ('02/09/1993', 15.3, 18.3),
                ('03/09/1993', 13.2, 18.3),
                ('04/09/1993', 12.8, 17.9),
                ('05/09/1993', 11.8, 17.4),
                ('06/09/1993', 12.9, 17.3),
                ('07/09/1993', 15.3, 17.0),
                ('08/09/1993', 17.3, 17.1),
                ('09/09/1993', 18.2, 17.5),
                ('10/09/1993', 17.4, 17.7),
                ('11/09/1993', 14.6, 17.9),
                ('12/09/1993', 16.7, 17.8),
                ('13/09/1993', 18.1, 17.9),
                ('14/09/1993', 18.2, 18.0),
                ('15/09/1993', 16.3, 18.0),
                ('16/09/1993', 16.1, 17.9),
                ('17/09/1993', 15.5, 17.7),
                ('18/09/1993', 14.1, 17.5),
                ('19/09/1993', 14.4, 17.1),
                ('20/09/1993', 15.9, 17.0),
                ('21/09/1993', 15.7, 17.0),
                ('22/09/1993', 15.2, 17.1),
                ('23/09/1993', 18.1, 17.0),
                ('24/09/1993', 17.6, 17.3),
                ('25/09/1993', 15.6, 17.4),
                ('26/09/1993', 12.7, 17.1),
                ('27/09/1993', 11.7, 16.7),
                ('28/09/1993', 11.6, 16.2),
                ('29/09/1993',  9.9, 15.8),
                ('30/09/1993',  9.9, 15.4),
                ('01/10/1993', 11.9, 15.1),
                ('02/10/1993', 15.5, 15.1),
                ('03/10/1993', 12.5, 15.3),
                ('04/10/1993', 12.4, 15.4),
                ('05/10/1993', 14.1, 15.4),
                ('06/10/1993', 15.3, 15.3),
                ('07/10/1993', 16.3, 15.5),
                ('08/10/1993', 15.7, 15.7),
                ('09/10/1993', 13.9, 15.8),
                ('10/10/1993', 14.2, 15.7),
                ('11/10/1993', 15.8, 15.5),
                ('12/10/1993', 17.7, 15.6),
                ('13/10/1993', 18.2, 15.8),
                ('14/10/1993', 15.6, 16.1),
                ('15/10/1993', 13.7, 15.9),
                ('16/10/1993', 13.7, 15.6),
                ('17/10/1993', 15.2, 15.4),
                ('18/10/1993',  8.3, 15.3),
                ('19/10/1993',  7.6, 14.6),
                ('20/10/1993',  8.1, 14.0),
                ('21/10/1993',  8.0, 13.4),
                ('22/10/1993',  7.0, 12.5),
                ('23/10/1993', 10.2, 11.8),
                ('24/10/1993',  9.1, 11.9),
                ('25/10/1993', 10.2, 12.1),
                ('26/10/1993',  6.7, 11.8),
                ('27/10/1993',  5.6, 11.8),
                ('28/10/1993',  5.3, 11.3),
                ('29/10/1993',  4.4, 10.7),
                ('30/10/1993',  4.3, 10.2),
                ('31/10/1993',  5.7,  9.9),
                ('01/11/1993',  6.9,  9.9),
                ('02/11/1993',  6.6,  9.9),
                ('03/11/1993', 10.2, 10.2),
                ('04/11/1993', 11.9, 10.6),
                ('05/11/1993', 10.0, 11.1),
                ('06/11/1993', 10.2, 11.1),
                ('07/11/1993', 11.1, 11.2),
                ('08/11/1993',  9.3, 11.3),
                ('09/11/1993',  9.1, 11.2),
                ('10/11/1993',  8.0, 11.1),
                ('11/11/1993',  8.5, 10.9),
                ('12/11/1993',  4.4, 10.8),
                ('13/11/1993', -1.9, 10.1),
                ('14/11/1993',  0.1,  8.8),
                ('15/11/1993',  1.2,  7.6),
                ('16/11/1993',  2.1,  7.0),
                ('17/11/1993',  1.5,  7.0),
                ('18/11/1993', -0.7,  6.8),
                ('19/11/1993', -3.9,  6.6),
                ('20/11/1993', -4.5,  5.8),
                ('21/11/1993', -3.0,  5.5),
                ('22/11/1993', -2.4,  5.4),
                ('23/11/1993', -2.5,  5.2),
                ('24/11/1993', -1.7,  5.1),
                ('25/11/1993', -0.9,  5.1),
                ('26/11/1993', -2.2,  5.0),
                ('27/11/1993', -3.7,  4.8),
                ('28/11/1993', -3.9,  4.8),
                ('29/11/1993', -3.7,  4.7),
                ('30/11/1993', -0.7,  4.7),
                ('01/12/1993', -1.7,  4.6),
                ('02/12/1993', -1.7,  4.5),
                ('03/12/1993', -3.2,  4.3),
                ('04/12/1993', -3.4,  4.4),
                ('05/12/1993', -1.3,  4.3),
                ('06/12/1993',  0.4,  4.2),
                ('07/12/1993', -0.9,  4.2),
                ('08/12/1993',  1.8,  4.1),
                ('09/12/1993',  7.0,  4.2),
                ('10/12/1993',  5.2,  4.1),
                ('11/12/1993',  5.1,  4.6),
                ('12/12/1993',  0.5,  4.9),
                ('13/12/1993',  3.0,  4.8),
                ('14/12/1993',  8.0,  4.9),
                ('15/12/1993',  3.8,  5.5),
                ('16/12/1993',  5.1,  5.6),
                ('17/12/1993',  0.2,  5.6),
                ('18/12/1993',  3.2,  5.3),
                ('19/12/1993',  5.0,  5.0),
                ('20/12/1993',  9.3,  5.2),
                ('21/12/1993', 12.3,  5.9),
                ('22/12/1993',  2.6,  6.3),
                ('23/12/1993',  2.0,  6.0),
                ('24/12/1993',  4.7,  5.5),
                ('25/12/1993',  1.0,  5.3),
                ('26/12/1993',  0.1,  4.8),
                ('27/12/1993',  0.3,  4.5),
                ('28/12/1993', -1.7,  4.3),
                ('29/12/1993', -4.7,  4.1),
                ('30/12/1993', -3.8,  3.9),
                ('31/12/1993',  0.8,  3.8)
             ]
            
            dates_long = ['31/07/1992', '06/08/1992', '18/08/1992', '26/08/1992', '03/08/1993', '21/08/1993']
            dates = valovi(data_long)
            
            if dates != dates_long:
                Check.error('Izraz valovi(<OBSEŽNI PODATKI>) vrne {} namesto {}', dates, dates_long)
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
        ] = "eyJwYXJ0IjozMzQ4NiwidXNlciI6MTA3NDh9:1uP4HN:vn1bgbhwdRkoefUJrc3HNkV39Fo6vgo8-iL6KXeZOcI"
        try:
            Check.equal(f'povprecja({data_short})', (-99.9, -99.9, -99.9))
            Check.equal(f'povprecja({data_supershort})', (10.0, 10.5, 11.0))
            
            avg_long = (22.7, 23.1, 23.4)
            avg = povprecja(data_long)
            
            if avg_long != avg:
                Check.error('Izraz povprecja(<OBSEŽNI PODATKI>) vrne {} namesto {}.', avg, avg_long)
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
