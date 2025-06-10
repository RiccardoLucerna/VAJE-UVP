# =============================================================================
# Življenja žab
#
# Ta naloga vsebuje 4 podnaloge. Pedagogom ni treba rešiti četrte, preostalim pa ne druge. Vseeno si preberite opis pri drugi, da boste lahko razumeli tretjo.
# =====================================================================@033483=
# 1. podnaloga
# Žabam v življenju ni lahko. Sobivajo s pitoni, zato se število žab $ž(t)$ in
# število pitonov $p(t)$ skozi leta $t$ giblje tako:
# 
# $$\begin{eqnarray*}
# \check{z}(t) &=& \check{z}(t - 1) + \alpha_1 \check{z}(t - 1) - 
#                  \alpha_2 \check{z}(t - 1) p(t - 1) \\
# p(t) &=& p(t - 1) + \alpha_3 p(t - 1) + \alpha_4 \check{z}(t - 1) p(t - 1)
# \end{eqnarray*}
# $$
# Pojasnilo: vsako leto vsaka žaba povzroči pojav $\alpha_1$ novih žabic.
# Število srečanj med žabami in pitoni v danem letu je sorazmerno s
# produktom teh dveh populacij, zato je $\alpha_2$
# stopnja tragičnosti teh srečanj (gledano z žabjega vidika).
# Pri pitonih je podobno,
# le da so zanje srečanja z žabami ... koristna.
# 
# Napišite funkcijo `cez_nekaj_let(alfe, zabe, pitoni, leta)`,
# ki sprejme četverico parametrov $\alpha_i$, trenutni populaciji žab in pitonov
# ter parameter `leta`, vrne pa populaciji žab in pitonov čez
# `leta` let. Funkcija naj ne zaokrožuje populacij na cela števila.
# 
#     >>> cez_nekaj_let((2.0, 0.1, 0.5, 0.01), 100, 5, 1)
#     (250.0, 12.5)
#     >>> cez_nekaj_let((0.1, 0.1, 0.5, 0.01), 5, 100, 1)
#     (0.0, 105.0)
# 
# Pazite! Negativnih populacij ni! V primeru izumrtja vrste, mora funkcija
# vrniti 0.
# =============================================================================
def cez_nekaj_let(alfe, zabe, pitoni, leta):
    alpha_1, alpha_2, alpha_3, alpha_4 = alfe
    z = zabe
    p = pitoni

    for _ in range(leta):
        novi_z = z + alpha_1 * z - alpha_2 * z * p
        novi_p = p + alpha_3 * p + alpha_4 * z * p

        # Če katera populacija pade pod 0, se vrne 0
        if novi_z < 0 or novi_p < 0:
            return 0

        z = novi_z
        p = novi_p

    return (z, p)

# =====================================================================@033493=
# 2. podnaloga
# **Naloga za pedagoge. Ostali jo preberite, reševati je pa ni treba.**
# 
# Vseeno se najde čas tudi za druženje in veselje, središče dogajanja pa je
# gostišče Pod Lokvanji, kjer strežejo eno samo specialiteto: muhe, zavite v
# lokvanjeve liste. Porcijo se da plačati bodisi s `p` prodniki (žabja valuta)
# bodisi z `l` listi lokvanja, ki jih gost prejme skupaj
# s prej naročenimi porcijami.
# 
# Zapišite funkcijo `neskoncno_porcij(g, p, l)`,
# ki vrne `True`, če lahko gost, ki pride v gostišče z `g`
# prodniki gotovine (in brez listov lokvanja), dobi neskončno porcij. Sicer
# naj vrne `False`.
# 
# Predpostavite lahko, da sta `p` in `l` pozitivni celi števili.
# 
#     >>> neskoncno_porcij(11, 2, 4)
#     False
#     >>> neskoncno_porcij(11, 1, 1)
#     True
# 
# V prvem primeru lahko gost z $11$ prodniki kupi $5$ porcij, nato pa
# še eno porcijo v zameno za $4$ liste. Na koncu v rokah drži en prodnik
# in dva lista, s čimer pa ne more dobiti nove porcije.
# =============================================================================

# =====================================================================@033481=
# 3. podnaloga
# Napišite funkcijo `stevilo_porcij(g, p, l)`, ki vrne število porcij, ki si
# jih lahko privošči žabec, ki pride v gostišče z `g` prodniki gotovine
# (in brez listov lokvanja).
# 
#     >>> stevilo_porcij(11, 2, 4)
#     6
# 
# Predpostavite lahko, da so vhodni podatki taki, da je število porcij končno.
# Če se vam ne sanja, o čem govori naloga, preberite navodilo prejšnje.
# =============================================================================

# =====================================================================@033482=
# 4. podnaloga
# **Pedagogom te naloge ni treba rešiti.**
# 
# Po naporni požrtiji se žabci odpravijo do svojih domov,
# pri čemer (iz že znanih razlogov) sledijo zemljevidu, na katerem so označena
# nevarna mesta. Zemljevid je podan kot seznam
# seznamov vrednosti `True` oz. `False`, kjer `True` pomeni, da je polje varno.
# Na zemljevidu
# 
#     [
#         [True,  True,  False, True],
#         [False, True,  False, True],
#         [False, True,  True,  False],
#         [True,  False, True,  True],
#     ]
# 
# je tako le ena varna pot med zgornjim levim poljem (gostišče)
# in spodnjim desnim poljem, saj se lahko žabci s skoki dolžine 1
# premikajo le v smeri dol ali desno. Napišite funkcijo
# `prestej_varne_poti(zemljevid, i, j)`,
# ki vrne število varnih poti od gostišča do $(i, j)$.
# 
#     >>> prestej_varne_poti([[True, False], [False, True]], 0, 0)
#     1
#     >>> prestej_varne_poti([[True, False], [False, True]], 1, 1)
#     0
#     >>> prestej_varne_poti([[True, True], [True, True]], 1, 1)
#     2
#     >>> prestej_varne_poti([[True, True, True], [True, True, True]], 1, 2)
#     3
# 
# Predpostavite lahko, da je začetno polje (gostišče) varno, tj.
# `zemljevid[0][0] == True` in je zato varna pot do njega ena sama.
# Če polje ni varno, ni varnih varnih poti do in preko njega.
# 
# **Namig:** rekurzija.
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
        ] = "eyJwYXJ0IjozMzQ4MywidXNlciI6MTA3NDh9:1uP4HN:vPlREB24FsPYwMqFukO0cMUsVgTXFtAHuYQvLcJwO7k"
        try:
            epsilon = 1e-6
            testni_primeri = [
                ((2.0, 0.1, 0.5, 0.01), 100, 5, 1),
                ((0.1, 0.1, 0.5, 0.01), 5, 100, 1),
            ]
            nabori_alf = [(2.0, 0.01, 0.01, 0.01), (2.0, 0.5, 0.5, 0.5), (1.5, 0.1, 0.01, 0.001)]
            pravilni_odgovori = [
                (250.0, 12.5),
                (0, 155.0),
                (29.0, 11.1),
                (83.781, 14.43),
                (239.2534017, 26.6638983),
                (653.9659213914215, 90.72482099157847),
                (1368.5883526459706, 684.9414807297882),
                (0, 10065.820223245812),
                (0, 10166.47842547827),
                (0, 10268.143209733053),
                (0, 10370.824641830382),
                (0, 65.0),
                (0, 97.5),
                (0, 146.25),
                (0, 219.375),
                (0, 329.0625),
                (0, 493.59375),
                (0, 740.390625),
                (0, 1110.5859375),
                (0, 1665.87890625),
                (15.0, 10.2),
                (22.200000000000003, 10.455),
                (32.2899, 10.791651),
                (45.878616837509995, 11.248028841624901),
                (63.092141553557965, 11.87655313544332),
                (82.79863672492064, 12.744635838387497),
                (101.47274451489652, 13.927320669745422),
                (112.35751607742984, 15.479837328540992),
                (106.96618304182024, 17.373911773343945),
                (28.0, 22.2),
                (77.784, 28.638),
                (211.07621808000002, 51.200161920000006),
                (525.1572888084278, 159.78352897077235),
                (736.3570177199464, 1000.4962129658169),
                (0, 8377.725251291567),
                (0, 8461.502503804482),
                (0, 8546.117528842527),
                (0, 8631.578704130952),
                (0, 130.0),
                (0, 195.0),
                (0, 292.5),
                (0, 438.75),
                (0, 658.125),
                (0, 987.1875),
                (0, 1480.78125),
                (0, 2221.171875),
                (0, 3331.7578125),
                (5.0, 20.4),
                (2.3000000000000007, 20.706),
                (0.9876199999999997, 20.960683799999998),
                (0.3989309465444002, 21.190991828534553),
                (0.15195312352380652, 21.411355489248272),
                (0.05453057426252872, 21.628722566486225),
                (0.018383769444781012, 21.846189218813205),
                (0.00579789302738027, 22.065052726307183),
                (0.0017016510333872307, 22.285831184385604),
                (27.0, 33.3),
                (72.009, 42.623999999999995),
                (185.33388384, 73.74335615999999),
                (419.33022547470813, 211.15221576689186),
                (372.56561395397455, 1098.6888003947106),
                (0, 5203.012363032769),
                (0, 5255.0424866630965),
                (0, 5307.592911529728),
                (0, 5360.668840645025),
                (0, 195.0),
                (0, 292.5),
                (0, 438.75),
                (0, 658.125),
                (0, 987.1875),
                (0, 1480.78125),
                (0, 2221.171875),
                (0, 3331.7578125),
                (0, 4997.63671875),
                (0, 30.6),
                (0, 30.906000000000002),
                (0, 31.21506),
                (0, 31.5272106),
                (0, 31.842482706),
                (0, 32.16090753306),
                (0, 32.4825166083906),
                (0, 32.8073417744745),
                (0, 33.13541519221925),
                (26.0, 44.4),
                (66.456, 56.388000000000005),
                (161.89479072, 94.42508928000001),
                (332.81507148297084, 248.23864084982918),
                (172.2696044561984, 1076.8966372510417),
                (0, 2942.831181018024),
                (0, 2972.259492828204),
                (0, 3001.982087756486),
                (0, 3032.0019086340512),
                (0, 260.0),
                (0, 390.0),
                (0, 585.0),
                (0, 877.5),
                (0, 1316.25),
                (0, 1974.375),
                (0, 2961.5625),
                (0, 4442.34375),
                (0, 6663.515625),
                (0, 40.8),
                (0, 41.208),
                (0, 41.62008),
                (0, 42.0362808),
                (0, 42.456643608),
                (0, 42.88121004408),
                (0, 43.3100221445208),
                (0, 43.74312236596601),
                (0, 44.18055358962567),
                (290.0, 20.1),
                (811.71, 78.59100000000001),
                (1797.1989939, 717.3079161),
                (0, 13615.931646575256),
                (0, 13752.09096304101),
                (0, 13889.61187267142),
                (0, 14028.507991398134),
                (0, 14168.793071312115),
                (0, 14310.481002025235),
                (0, 515.0),
                (0, 772.5),
                (0, 1158.75),
                (0, 1738.125),
                (0, 2607.1875),
                (0, 3910.78125),
                (0, 5866.171875),
                (0, 8799.2578125),
                (0, 13198.88671875),
                (150.0, 11.1),
                (208.5, 12.876),
                (252.78539999999998, 15.689405999999998),
                (235.35822285275998, 19.8123528314724),
                (122.09554183718086, 24.673476512734315),
                (3.9867062100269663, 27.93273276169091),
                (0, 28.323419688471876),
                (0, 28.606653885356597),
                (0, 28.89272042421016),
                (280.0, 40.2),
                (727.4399999999999, 153.16200000000003),
                (1068.1583471999995, 1268.8552728000002),
                (0, 14834.927335828528),
                (0, 14983.276609186814),
                (0, 15133.109375278682),
                (0, 15284.44046903147),
                (0, 15437.284873721785),
                (0, 15591.657722459002),
                (0, 1030.0),
                (0, 1545.0),
                (0, 2317.5),
                (0, 3476.25),
                (0, 5214.375),
                (0, 7821.5625),
                (0, 11732.34375),
                (0, 17598.515625),
                (0, 26397.7734375),
                (50.0, 22.2),
                (14.0, 23.532),
                (2.0551999999999992, 24.096768),
                (0.1856322406399995, 24.387259357593603),
                (0.011374441838109206, 24.635659012767157),
                (0.0004144175368519243, 24.882295819765414),
                (4.87786764499917e-06, 25.131129089622814),
                (0, 25.382440503105364),
                (0, 25.63626490813642),
                (270.0, 60.3),
                (647.19, 223.713),
                (493.72183530000007, 1673.7982947),
                (0, 9954.443937459942),
                (0, 10053.988376834543),
                (0, 10154.528260602889),
                (0, 10256.073543208917),
                (0, 10358.634278641006),
                (0, 10462.220621427416),
                (0, 1545.0),
                (0, 2317.5),
                (0, 3476.25),
                (0, 5214.375),
                (0, 7821.5625),
                (0, 11732.34375),
                (0, 17598.515625),
                (0, 26397.7734375),
                (0, 39596.66015625),
                (0, 33.3),
                (0, 33.632999999999996),
                (0, 33.96932999999999),
                (0, 34.30902329999999),
                (0, 34.65211353299999),
                (0, 34.99863466832999),
                (0, 35.34862101501329),
                (0, 35.702107225163424),
                (0, 36.05912829741506),
                (260.0, 80.4),
                (570.96, 290.244),
                (55.70285759999979, 1950.3235824000003),
                (0, 3056.2127860674873),
                (0, 3086.7749139281623),
                (0, 3117.642663067444),
                (0, 3148.8190896981187),
                (0, 3180.3072805950997),
                (0, 3212.110353401051),
                (0, 2060.0),
                (0, 3090.0),
                (0, 4635.0),
                (0, 6952.5),
                (0, 10428.75),
                (0, 15643.125),
                (0, 23464.6875),
                (0, 35197.03125),
                (0, 52795.546875),
                (0, 44.4),
                (0, 44.844),
                (0, 45.29244),
                (0, 45.7453644),
                (0, 46.202818044),
                (0, 46.66484622444),
                (0, 47.1314946866844),
                (0, 47.60280963355124),
                (0, 48.07883772988675),
                (2900.0, 110.1),
                (5507.1, 3304.1009999999997),
                (0, 185297.288181),
                (0, 187150.26106281002),
                (0, 189021.7636734381),
                (0, 190911.9813101725),
                (0, 192821.10112327422),
                (0, 194749.31213450697),
                (0, 196696.80525585203),
                (0, 5015.0),
                (0, 7522.5),
                (0, 11283.75),
                (0, 16925.625),
                (0, 25388.4375),
                (0, 38082.65625),
                (0, 57123.984375),
                (0, 85685.9765625),
                (0, 128528.96484375),
                (1500.0, 20.1),
                (735.0, 50.45100000000001),
                (0, 88.03699500000002),
                (0, 88.91736495000002),
                (0, 89.80653859950002),
                (0, 90.70460398549501),
                (0, 91.61165002534996),
                (0, 92.52776652560345),
                (0, 93.45304419085949),
                (2800.0, 220.2),
                (2234.4000000000005, 6388.0019999999995),
                (0, 149185.39870800002),
                (0, 150677.25269508),
                (0, 152184.02522203082),
                (0, 153705.86547425113),
                (0, 155242.92412899365),
                (0, 156795.35337028358),
                (0, 158363.30690398643),
                (0, 10030.0),
                (0, 15045.0),
                (0, 22567.5),
                (0, 33851.25),
                (0, 50776.875),
                (0, 76165.3125),
                (0, 114247.96875),
                (0, 171371.953125),
                (0, 257057.9296875),
                (500.0, 40.2),
                (0, 60.702000000000005),
                (0, 61.309020000000004),
                (0, 61.922110200000006),
                (0, 62.541331302),
                (0, 63.16674461502),
                (0, 63.798412061170204),
                (0, 64.4363961817819),
                (0, 65.08076014359972),
                (2700.0, 330.3),
                (0, 9251.703),
                (0, 9344.22003),
                (0, 9437.6622303),
                (0, 9532.038852603),
                (0, 9627.35924112903),
                (0, 9723.63283354032),
                (0, 9820.869161875722),
                (0, 9919.077853494478),
                (0, 15045.0),
                (0, 22567.5),
                (0, 33851.25),
                (0, 50776.875),
                (0, 76165.3125),
                (0, 114247.96875),
                (0, 171371.953125),
                (0, 257057.9296875),
                (0, 385586.89453125),
                (0, 60.3),
                (0, 60.903),
                (0, 61.512029999999996),
                (0, 62.1271503),
                (0, 62.748421803),
                (0, 63.37590602103),
                (0, 64.0096650812403),
                (0, 64.64976173205271),
                (0, 65.29625934937323),
                (2600.0, 440.4),
                (0, 11895.204),
                (0, 12014.15604),
                (0, 12134.2976004),
                (0, 12255.640576404),
                (0, 12378.19698216804),
                (0, 12501.97895198972),
                (0, 12626.998741509618),
                (0, 12753.268728924715),
                (0, 20060.0),
                (0, 30090.0),
                (0, 45135.0),
                (0, 67702.5),
                (0, 101553.75),
                (0, 152330.625),
                (0, 228495.9375),
                (0, 342743.90625),
                (0, 514115.859375),
                (0, 80.4),
                (0, 81.20400000000001),
                (0, 82.01604),
                (0, 82.83620040000001),
                (0, 83.66456240400001),
                (0, 84.50120802804001),
                (0, 85.34622010832041),
                (0, 86.19968230940361),
                (0, 87.06167913249764),
                (29000.0, 1010.1),
                (0, 293949.201),
                (0, 296888.69301),
                (0, 299857.57994009997),
                (0, 302856.15573950094),
                (0, 305884.717296896),
                (0, 308943.56446986494),
                (0, 312033.0001145636),
                (0, 315153.3301157093),
                (0, 50015.0),
                (0, 75022.5),
                (0, 112533.75),
                (0, 168800.625),
                (0, 253200.9375),
                (0, 379801.40625),
                (0, 569702.109375),
                (0, 854553.1640625),
                (0, 1281829.74609375),
                (15000.0, 110.1),
                (0, 1762.701),
                (0, 1780.32801),
                (0, 1798.1312901),
                (0, 1816.1126030009998),
                (0, 1834.27372903101),
                (0, 1852.61646632132),
                (0, 1871.1426309845333),
                (0, 1889.8540572943787),
                (28000.0, 2020.2),
                (0, 567696.402),
                (0, 573373.36602),
                (0, 579107.0996802),
                (0, 584898.1706770019),
                (0, 590747.152383772),
                (0, 596654.6239076097),
                (0, 602621.1701466858),
                (0, 608647.3818481526),
                (0, 100030.0),
                (0, 150045.0),
                (0, 225067.5),
                (0, 337601.25),
                (0, 506401.875),
                (0, 759602.8125),
                (0, 1139404.21875),
                (0, 1709106.328125),
                (0, 2563659.4921875),
                (5000.0, 220.2),
                (0, 1323.402),
                (0, 1336.6360200000001),
                (0, 1350.0023802),
                (0, 1363.502404002),
                (0, 1377.13742804202),
                (0, 1390.9088023224401),
                (0, 1404.8178903456646),
                (0, 1418.8660692491212),
                (27000.0, 3030.3),
                (0, 821241.603),
                (0, 829454.01903),
                (0, 837748.5592202999),
                (0, 846126.0448125029),
                (0, 854587.305260628),
                (0, 863133.1783132342),
                (0, 871764.5100963666),
                (0, 880482.1551973303),
                (0, 150045.0),
                (0, 225067.5),
                (0, 337601.25),
                (0, 506401.875),
                (0, 759602.8125),
                (0, 1139404.21875),
                (0, 1709106.328125),
                (0, 2563659.4921875),
                (0, 3845489.23828125),
                (0, 330.3),
                (0, 333.603),
                (0, 336.93903),
                (0, 340.3084203),
                (0, 343.711504503),
                (0, 347.14861954803),
                (0, 350.6201057435103),
                (0, 354.1263068009454),
                (0, 357.66756986895484),
                (26000.0, 4040.4),
                (0, 1054584.804),
                (0, 1065130.65204),
                (0, 1075781.9585604002),
                (0, 1086539.778146004),
                (0, 1097405.1759274642),
                (0, 1108379.2276867388),
                (0, 1119463.0199636063),
                (0, 1130657.6501632424),
                (0, 200060.0),
                (0, 300090.0),
                (0, 450135.0),
                (0, 675202.5),
                (0, 1012803.75),
                (0, 1519205.625),
                (0, 2278808.4375),
                (0, 3418212.65625),
                (0, 5127318.984375),
                (0, 440.4),
                (0, 444.804),
                (0, 449.25203999999997),
                (0, 453.74456039999995),
                (0, 458.282006004),
                (0, 462.86482606403996),
                (0, 467.49347432468034),
                (0, 472.16840906792714),
                (0, 476.8900931586064),
            ]
            for z0 in [10, 100, 1000, 10000]:
                for p0 in [10, 20, 30, 40]:
                    for alfe in nabori_alf:
                        for leto in range(1, 10):
                            testni_primeri.append((alfe, z0, p0, leto))
            
            
            def preveri_vse_prva():
                for vhod, pravilni_odgovor in zip(testni_primeri, pravilni_odgovori):
                    klic = f"cez_nekaj_let({vhod[0]}, {vhod[1]}, {vhod[2]}, {vhod[3]})"
                    dejanski_odgovor = cez_nekaj_let(*vhod)
                    try:
                        assert isinstance(dejanski_odgovor, tuple) and len(dejanski_odgovor) == 2
                        napaka = max(abs(x - y) for x, y in zip(dejanski_odgovor, pravilni_odgovor))
                    except:
                        Check.error(
                            f"Klic {klic} bi moral vrniti par (zabe, pitoni), vrnil pa je {dejanski_odgovor}."
                        )
                        break
                    if napaka > epsilon:
                        Check.error(
                            f"Pravilni odgovor {pravilni_odgovor} in vaš odgovor {dejanski_odgovor} "
                            f"po klicu {klic} "
                            f"se na vsaj eni od komponent razlikujeta za več kot {epsilon}."
                        )
                        break
            
            
            preveri_vse_prva()
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
        ] = "eyJwYXJ0IjozMzQ5MywidXNlciI6MTA3NDh9:1uP4HN:RBInsaNS9DUXidcazt-m9Swu5AGx_CnyiCllrxvoZm8"
        try:
            Check.equal('neskoncno_porcij(11, 2, 4)', False)
            Check.equal('neskoncno_porcij(11, 1, 1)', True)
            Check.equal('neskoncno_porcij(1, 2, 1)', False)
            Check.equal('neskoncno_porcij(2, 2, 1)', True)
            Check.equal('neskoncno_porcij(3, 2, 1)', True)
            Check.equal('neskoncno_porcij(10 ** 10, 1, 2)', False)
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
        ] = "eyJwYXJ0IjozMzQ4MSwidXNlciI6MTA3NDh9:1uP4HN:EIK_BExRVtHcKJKyOK7JTY_QljnnmyRV-tWfZrSOLPM"
        try:
            def porcije_drugace(g, p, l, vse_porcije=0, lokvanji=0):
                if g >= p:
                    return porcije_drugace(g - p, p, l, vse_porcije + 1, lokvanji + 1)
                elif lokvanji >= l:
                    return porcije_drugace(g, p, l, vse_porcije + 1, lokvanji - l + 1)
                else:
                    return vse_porcije
            
            
            def preveri_vse_druga():
                for gotovina in range(100):
                    for porcija in range(1, 11):
                        for lokvanj in range(2, 11):
                            pravilno = porcije_drugace(gotovina, porcija, lokvanj)
                            if not Check.equal(
                                f"stevilo_porcij({gotovina}, {porcija}, {lokvanj})", pravilno
                            ):
                                return
            
            
            preveri_vse_druga()
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
        ] = "eyJwYXJ0IjozMzQ4MiwidXNlciI6MTA3NDh9:1uP4HN:4G4e2K-EybHgq1uWeKQJJD4Np-B9epJ-eUKauErag1Y"
        try:
            zemljevid_iz_primera = [
                [True, True, False, True],
                [False, True, False, True],
                [False, True, True, False],
                [True, False, True, True],
            ]
            zanimiv_zemljevid = [
                [True, True, True, True, True, True, True],
                [True, False, False, True, False, False, True],
                [True, False, True, True, True, True, False],
                [True, False, True, True, True, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, False, True],
            ]
            testni_primeri = [
                (([[True, False], [False, True]], 0, 0), 1),
                (([[True, False], [False, True]], 1, 1), 0),
                (([[True, True], [True, True]], 1, 1), 2),
                (([[True, True, True], [True, True, True]], 1, 2), 3),
            ]
            testni_primeri.extend(
                [
                    ((zemljevid_iz_primera, i, j), 1)
                    for i, j in [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
                ]
            )
            testni_primeri.extend(
                [((zemljevid_iz_primera, i, j), 0) for i, j in [(0, 3), (1, 3), (3, 0)]]
            )
            resitve_zanimiv = [
                [1, 1, 1, 1, 1, 1, 1],
                [1, None, None, 1, None, None, 1],
                [1, None, 0, 1, 1, 1, None],
                [1, None, 0, 1, 2, None, 0],
                [1, 1, 1, 2, 4, 4, 4],
                [1, 2, 3, 5, 9, None, 4],
            ]
            for i in range(6):
                for j in range(7):
                    if resitve_zanimiv[i][j] is None:
                        continue
                    testni_primeri.append(((zanimiv_zemljevid, i, j), resitve_zanimiv[i][j]))
            
            
            def preveri_vse_tretja():
                Check.equal("prestej_varne_poti([[True, False], [False, False]], 0, 0)", 1)
                Check.equal("prestej_varne_poti([[True, True], [True, True]], 1, 1)", 2)
                for aguamenti, odgovor in testni_primeri:
                    z, i, j = aguamenti
                    if not Check.equal(f"prestej_varne_poti({z}, {i}, {j})", odgovor):
                        return
            
            
            preveri_vse_tretja()
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
