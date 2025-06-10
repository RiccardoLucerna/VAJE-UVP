# =============================================================================
# Pesnik France
#
# France zdaj obvlada iskanje rim, slovensko abecedo in osnove Pythona. 
# Končno se lahko loti pisanja.
# =====================================================================@040225=
# 1. podnaloga
# Nekatere pesmi imajo predpisano število kitic in vrstic v njej.
# Zapišite funkcijo `dolzine_kitic(pesem)`, ki vrne seznam dolžin kitic v
# nizu `pesem`. Za vhoda
# 
#     pesem1 = """Gnoj je zlato,    |    pesem2 = """Nina,
#     zlato je gnoj!                |    Nina,
#                                   |    Nina,
#     Ti si sova,                   |    ena in edina."""
#     jaz pa noj."""                |
# 
# velja, da je `dolzine_kitic(pesem1) == [2, 2]` in `dolzine_kitic(pesem2) == [4]`.
# =============================================================================
def dolzine_kitic(pesem):
    # Split the poem into stanzas (separated by empty lines)
    stanzas = []
    current_stanza = []
    
    for line in pesem.split('\n'):
        stripped_line = line.strip()
        if stripped_line:  # If line is not empty
            current_stanza.append(stripped_line)
        elif current_stanza:  # If we hit an empty line and have a current stanza
            stanzas.append(current_stanza)
            current_stanza = []
    
    # Add the last stanza if there's any
    if current_stanza:
        stanzas.append(current_stanza)
    
    # Return the lengths of each stanza
    return [len(stanza) for stanza in stanzas]
# =====================================================================@040224=
# 2. podnaloga
# Veliko pesmi ima predpisano število zlogov v verzu. Napiši funkcijo
# `stevilo_zlogov(verz)`, ki vrne seznam števil zlogov po besedah v verzu.
# Pri tem ignorirajte enočrkovne besede (npr. _v_, _k_, _s_).
# 
# Število zlogov v besedi dobimo tako, da preštejemo samoglasnike in
# črke `r`, ki ne stojijo ob soglasniku:
# 
#     >>> stevilo_zlogov("Rdečo mašno maš v laseh")
#     [3, 2, 1, 2]
#     >>> stevilo_zlogov("Jaz sem hrast")
#     [1, 1, 1]
# =============================================================================
def stevilo_zlogov(verz):
    samoglasnik = {'a', 'e', 'i', 'o', 'u'}
    soglasnik = {'b', 'c', 'č', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
                 'n', 'p', 'r', 's', 'š', 't', 'v', 'z', 'ž'}
    
    sez = []
    for beseda in verz.split():
        # ignorirajte enočrkovne besede
        if len(beseda) <= 1:
            continue
        
        count = 0
        beseda = beseda.lower()  # imam tako definirane crke
        for i, crka in enumerate(beseda):
            # prestej samoglasnike
            if crka in samoglasnik:
                count += 1
            # prestej 'r' ki ne stojijo ob soglasniku
            elif crka == 'r':
                prejsnji = beseda[i-1] in soglasnik if i > 0 else False
                naslednji = beseda[i+1] in soglasnik if i < len(beseda)-1 else False
                if not (prejsnji and naslednji):
                    count += 1
        
        if count > 0:
            sez.append(count)
    
    return sez
# =====================================================================@040226=
# 3. podnaloga
# Veliko pesmi ima predpisan tudi ritem, torej morajo biti poudarjeni zlogi
# na pravih mestih. Ta'ke zlo'ge bo'mo pri' te'j nalo'gi ozna'čili z eno'jnim
# narekova'jem ti'k za' naglaše'nim sa'mogla'snikom ozi'roma r'jem.
# 
# Zapišite funkcijo `ali_je_amfibrah(verz)`, ki vrne `True`, če je `verz`
# zapisan v amfibrahu, in `False` sicer. Če poudarjene zloge označimo s `P`,
# nepoudarjene pa z `N`, potem je verz v amfibrahu natanko tedaj, ko
# mu pripada zaporedje zlogov `NPN NPN ... NPN`.
# 
#     >>> ali_je_ambfibrah("pole'tje")
#     True
#     >>> ali_je_amfibrah("V pole'tno nebo' poleti'jo sini'ce")
#     True
# 
# Predpostavite lahko, da verz vsebuje le črke, presledke in enojne narekovaje.
# Beseda ima lahko več kot en poudarjen zlog.
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
        ] = "eyJwYXJ0Ijo0MDIyNSwidXNlciI6MTA3NDh9:1uNs0I:_Ar3FY9W2mc8qzx1HMpSyFww-vQHUgH_2xvA1hBLlMs"
        try:
            def preveri_dv():
                Check.equal("dolzine_kitic('Gnoj je zlato,\\nzlato je gnoj\\n\\nTi si sova,\\njaz pa noj.')", [2, 2])
                Check.equal("dolzine_kitic('Nina,\\nNina\\nNina,\\nena in edina\\n\\n\\n')", [4])
                Check.equal("dolzine_kitic('')", [])
                Check.equal("dolzine_kitic('a')", [1])
                Check.equal("dolzine_kitic('\\n\\na\\n\\n\\nb\\nc\\nd\\n\\n')", [1, 3])
                
            
            preveri_dv()
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
        ] = "eyJwYXJ0Ijo0MDIyNCwidXNlciI6MTA3NDh9:1uNs0I:sg--R_fPpnBuDYZWcLpx2JjuBh90KKg-acC0LCLfFgw"
        try:
            def preveri_sz():
                Check.equal("stevilo_zlogov('Rdečo mašno maš v laseh')", [3, 2,  1, 2])
                Check.equal("stevilo_zlogov('Jaz sem hrast')", [1, 1, 1])
                Check.equal("stevilo_zlogov('Razžvrkljati, godrnjati - posesati')", [4, 4, 4])
                Check.equal("stevilo_zlogov('Šmrkelj, šmrklja')", [2, 2])
                Check.equal("stevilo_zlogov('')", [])
                Check.equal("stevilo_zlogov('a')", [])
                Check.equal("stevilo_zlogov('a b c d e f g h i j k l m n o p q r s t u v w x y z')", [])
                Check.equal("stevilo_zlogov('Paleolitik daleč je za nami')", [5, 2, 1, 1, 2])
                Check.equal("stevilo_zlogov('Aaaa Eeee Iiii Oooo Uuuu')", [4, 4, 4, 4, 4])
                Check.equal("stevilo_zlogov('Prav, pokr je najslabš.')", [1, 2, 1, 2])
                Check.equal("stevilo_zlogov('Park za pet mark')", [1, 1, 1, 1])
                Check.equal("stevilo_zlogov('Tri, tir, Rit, Rotterdam')", [1, 1, 1, 3])
            
            
            preveri_sz()
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
        ] = "eyJwYXJ0Ijo0MDIyNiwidXNlciI6MTA3NDh9:1uNs0I:lwp9mmGGc7z6v2bxX-Zm5FTatT_tYxYo-fQ4-i7kOOw"
        try:
            def preveri_am():
                Check.equal("ali_je_amfibrah('pole\\'tje')", True)
                Check.equal("ali_je_amfibrah('Ljublja\\'na Tira\\'na Alba\\'ni')", True)
                Check.equal("ali_je_amfibrah('Ljublja\\'na Tira\\'na Alba\\'nija')", False)
                Check.equal("ali_je_amfibrah('Lja\\'na Tira\\'na Alba\\'ni')", False)
                Check.equal("ali_je_amfibrah('Lja\\'na Tira\\'na Alba\\'nija')", False)
                Check.equal("ali_je_amfibrah('V pole\\'tno nebo\\' poleti\\'jo sini\\'ce')", True)
                Check.equal("ali_je_amfibrah('aa\\'aaa\\'aaa\\'aaa\\'a')", True)
                Check.equal("ali_je_amfibrah('aa\\' aaa\\'aa a\\'aaa\\'a')", True)
                Check.equal("ali_je_amfibrah('aa\\'aa a\\'aaa\\'a aa\\'a')", True)
            
            preveri_am()
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
