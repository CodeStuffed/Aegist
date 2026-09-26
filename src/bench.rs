//! `aegist eval`: how good the model really is, measured by running code.
//!
//! Each problem is a Python function's signature and docstring; the model
//! writes the body, and the problem's tests decide - no opinions involved.
//! It reports pass@1 (its careful answer) and pass@k (any of k tries), and
//! how well its own verdicts predict the tests: the "does it know when it's
//! wrong" number.

use crate::brain::{Brain, GenOptions};
use crate::config::Settings;
use crate::corpus::FILE;
use crate::proc;
use crate::verify::{self, Confidence, Names, Status, Verdict};
use anyhow::{bail, Result};
use std::time::Duration;

pub struct Problem {
    pub name: &'static str,
    pub prompt: &'static str,
    pub tests: &'static str,
}

macro_rules! problem {
    ($name:expr, $prompt:expr, $tests:expr) => {
        Problem { name: $name, prompt: $prompt, tests: $tests }
    };
}

pub const PROBLEMS: &[Problem] = &[
    problem!("add", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"),
    problem!("is_even", "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n", "assert is_even(4)\nassert not is_even(7)\nassert is_even(0)\n"),
    problem!("factorial", "def factorial(n):\n    \"\"\"Return n! (the product 1 * 2 * ... * n); factorial(0) is 1.\"\"\"\n",
             "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800\n"),
    problem!("fibonacci", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number: fibonacci(0) = 0, fibonacci(1) = 1.\"\"\"\n",
             "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\n"),
    problem!("reverse_string", "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"\n",
             "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n"),
    problem!("is_palindrome", "def is_palindrome(s):\n    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n",
             "assert is_palindrome('racecar')\nassert not is_palindrome('abc')\nassert is_palindrome('')\n"),
    problem!("count_vowels", "def count_vowels(s):\n    \"\"\"Return how many vowels (a, e, i, o, u, either case) are in s.\"\"\"\n",
             "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU xyz') == 5\nassert count_vowels('') == 0\n"),
    problem!("max_of_list", "def max_of_list(xs):\n    \"\"\"Return the largest number in the non-empty list xs.\"\"\"\n",
             "assert max_of_list([3, 9, 2]) == 9\nassert max_of_list([-5, -2]) == -2\n"),
    problem!("sum_of_squares", "def sum_of_squares(n):\n    \"\"\"Return 1*1 + 2*2 + ... + n*n.\"\"\"\n",
             "assert sum_of_squares(1) == 1\nassert sum_of_squares(3) == 14\nassert sum_of_squares(0) == 0\n"),
    problem!("is_prime", "def is_prime(n):\n    \"\"\"Return True if n is a prime number, else False.\"\"\"\n",
             "assert is_prime(2) and is_prime(13) and is_prime(97)\nassert not is_prime(1) and not is_prime(0) and not is_prime(91)\n"),
    problem!("gcd", "def gcd(a, b):\n    \"\"\"Return the greatest common divisor of the positive integers a and b.\"\"\"\n",
             "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(10, 10) == 10\n"),
    problem!("flatten", "def flatten(lists):\n    \"\"\"Return one list with the items of every list in lists, in order.\"\"\"\n",
             "assert flatten([[1, 2], [3], []]) == [1, 2, 3]\nassert flatten([]) == []\n"),
    problem!("unique_sorted", "def unique_sorted(xs):\n    \"\"\"Return the distinct items of xs in ascending order.\"\"\"\n",
             "assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]\nassert unique_sorted([]) == []\n"),
    problem!("word_count", "def word_count(s):\n    \"\"\"Return the number of words in s (words are separated by whitespace).\"\"\"\n",
             "assert word_count('the quick  brown fox') == 4\nassert word_count('') == 0\n"),
    problem!("celsius_to_fahrenheit", "def celsius_to_fahrenheit(c):\n    \"\"\"Convert a temperature from Celsius to Fahrenheit.\"\"\"\n",
             "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\n"),
    problem!("running_sum", "def running_sum(xs):\n    \"\"\"Return the list of running totals of xs: [x0, x0 + x1, ...].\"\"\"\n",
             "assert running_sum([1, 2, 3]) == [1, 3, 6]\nassert running_sum([]) == []\n"),
    problem!("char_frequency", "def char_frequency(s):\n    \"\"\"Return a dict mapping each character of s to how many times it appears.\"\"\"\n",
             "assert char_frequency('aab') == {'a': 2, 'b': 1}\nassert char_frequency('') == {}\n"),
    problem!("merge_sorted", "def merge_sorted(a, b):\n    \"\"\"Merge the sorted lists a and b into one sorted list.\"\"\"\n",
             "assert merge_sorted([1, 4], [2, 3, 5]) == [1, 2, 3, 4, 5]\nassert merge_sorted([], [1]) == [1]\n"),
    problem!("binary_search", "def binary_search(xs, target):\n    \"\"\"Return the index of target in the sorted list xs, or -1 if it isn't there.\"\"\"\n",
             "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5, 7], 4) == -1\nassert binary_search([], 1) == -1\n"),
    problem!("fizzbuzz", "def fizzbuzz(n):\n    \"\"\"Return the FizzBuzz strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else the number.\"\"\"\n",
             "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\n"),
    problem!("transpose", "def transpose(matrix):\n    \"\"\"Return the transpose of a matrix given as a list of equal-length rows.\"\"\"\n",
             "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"),
    problem!("remove_duplicates", "def remove_duplicates(xs):\n    \"\"\"Return xs without repeated items, keeping the first of each, in order.\"\"\"\n",
             "assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]\nassert remove_duplicates([]) == []\n"),
    problem!("is_anagram", "def is_anagram(a, b):\n    \"\"\"Return True if a and b contain the same letters the same number of times.\"\"\"\n",
             "assert is_anagram('listen', 'silent')\nassert not is_anagram('abc', 'abd')\n"),
    problem!("capitalize_words", "def capitalize_words(s):\n    \"\"\"Return s with the first letter of every space-separated word in upper case.\"\"\"\n",
             "assert capitalize_words('hello big world') == 'Hello Big World'\n"),
    problem!("second_largest", "def second_largest(xs):\n    \"\"\"Return the second largest distinct number in xs (which has at least two distinct numbers).\"\"\"\n",
             "assert second_largest([4, 1, 4, 3]) == 3\nassert second_largest([1, 2]) == 1\n"),
];

/// Stop a Python function's body at the next line that starts at the left
/// margin (the next top-level statement).
pub fn end_of_function(text: &str) -> Option<usize> {
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let top_level = !line.starts_with([' ', '\t', '\n', '\r']) && !line.trim().is_empty();
        if top_level && offset > 0 && line.ends_with('\n') {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

pub struct Outcome {
    pub name: &'static str,
    /// Per candidate: (tests passed, Aegist's verdict).
    pub candidates: Vec<(bool, Verdict)>,
}

pub struct Summary {
    pub problems: usize,
    pub pass_at_1: usize,
    pub pass_at_k: usize,
    pub k: usize,
    /// Candidates Aegist stood behind: how many passed their tests.
    pub accepted: (usize, usize),
    /// Candidates Aegist refused: how many failed their tests.
    pub refused: (usize, usize),
}

pub fn summarize(outcomes: &[Outcome], k: usize) -> Summary {
    let mut s = Summary { problems: outcomes.len(), pass_at_1: 0, pass_at_k: 0, k, accepted: (0, 0), refused: (0, 0) };
    for o in outcomes {
        if o.candidates.first().is_some_and(|c| c.0) {
            s.pass_at_1 += 1;
        }
        if o.candidates.iter().any(|c| c.0) {
            s.pass_at_k += 1;
        }
        for &(passed, verdict) in &o.candidates {
            if verdict > Verdict::Refused {
                s.accepted.1 += 1;
                s.accepted.0 += passed as usize;
            } else {
                s.refused.1 += 1;
                s.refused.0 += !passed as usize;
            }
        }
    }
    s
}

/// Run every problem: `k` candidates each (the first careful, the rest sampled).
pub fn run(brain: &Brain, settings: &Settings, k: usize, names: &Names, log: &mut dyn FnMut(&Outcome, usize)) -> Result<Vec<Outcome>> {
    let Some(python) = proc::python() else { bail!("the benchmark runs Python tests, and Python isn't installed") };
    let py = crate::lang::by_name("python");
    let mut outcomes = Vec::new();
    for (i, p) in PROBLEMS.iter().enumerate() {
        let prompt = format!("{FILE}eval/{}.py\n{}", p.name, p.prompt);
        let opts: Vec<GenOptions> = (0..k.max(1))
            .map(|j| GenOptions { max_new: 200, temperature: if j == 0 { 0.0 } else { 0.8 }, top_p: 0.95, seed: 1000 + j as u64,
                                  speculative: true })
            .collect();
        let gens = brain.generate_many(&prompt, &opts, &end_of_function);
        let mut candidates = Vec::new();
        for g in &gens {
            let code = format!("{}{}", p.prompt, g.text);
            let program = format!("{code}\n\n{}", p.tests);
            let dir = std::env::temp_dir().join(format!("aegist-eval-{}-{}", std::process::id(), i));
            std::fs::create_dir_all(&dir)?;
            let file = dir.join("t.py");
            std::fs::write(&file, &program)?;
            let mut cmd = std::process::Command::new(python);
            cmd.arg(&file).current_dir(&dir);
            let passed = proc::run(cmd, Some(Duration::from_secs(10)), None).is_ok_and(|o| o.ok());
            let _ = std::fs::remove_dir_all(&dir);
            // Aegist's own verdict, without running the tests
            let syntax = match py {
                Some(l) => verify::syntax(l, &code, "t.py", settings.honesty.check_timeout_s),
                None => Status::Skipped("no python".into()),
            };
            let invented = verify::invented_names(&g.text, py, p.prompt, names);
            let conf = Confidence::of(g, brain.typical_nll(), &settings.honesty);
            let verdict = verify::judge(syntax, invented, Status::Skipped("not run".into()), conf, &settings.honesty).verdict;
            candidates.push((passed, verdict));
        }
        let o = Outcome { name: p.name, candidates };
        log(&o, i);
        outcomes.push(o);
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_bodies_end_at_the_next_top_level_line() {
        assert_eq!(end_of_function("    return a + b\n\n\ndef other():\n"), Some(19));
        assert_eq!(end_of_function("    x = 1\n    return x\n"), None);
        assert_eq!(end_of_function("    return 1\nprint(f())\n"), Some(13));
    }

    #[test]
    fn every_problem_is_solvable_and_its_tests_are_real() {
        // reference solutions: the tests must pass with them and fail without
        let Some(python) = proc::python() else { return };
        let solutions = [
            "    return a + b\n", "    return n % 2 == 0\n", "    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
            "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n", "    return s[::-1]\n", "    return s == s[::-1]\n",
            "    return sum(c in 'aeiouAEIOU' for c in s)\n", "    return max(xs)\n", "    return sum(i * i for i in range(1, n + 1))\n",
            "    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n",
            "    while b:\n        a, b = b, a % b\n    return a\n", "    return [x for l in lists for x in l]\n", "    return sorted(set(xs))\n",
            "    return len(s.split())\n", "    return c * 9 / 5 + 32\n",
            "    out, t = [], 0\n    for x in xs:\n        t += x\n        out.append(t)\n    return out\n",
            "    d = {}\n    for ch in s:\n        d[ch] = d.get(ch, 0) + 1\n    return d\n", "    return sorted(a + b)\n",
            "    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if xs[m] == target:\n            return m\n        if xs[m] < target:\n            lo = m + 1\n        else:\n            hi = m - 1\n    return -1\n",
            "    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]\n",
            "    return [list(r) for r in zip(*matrix)]\n", "    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return seen\n",
            "    return sorted(a) == sorted(b)\n", "    return ' '.join(w[:1].upper() + w[1:] for w in s.split(' '))\n",
            "    return sorted(set(xs))[-2]\n",
        ];
        assert_eq!(solutions.len(), PROBLEMS.len());
        let tmp = tempfile::tempdir().unwrap();
        for (p, sol) in PROBLEMS.iter().zip(solutions) {
            for (body, should_pass) in [(sol, true), ("    return None\n", false)] {
                let file = tmp.path().join(format!("{}.py", p.name));
                std::fs::write(&file, format!("{}{body}\n\n{}", p.prompt, p.tests)).unwrap();
                let mut cmd = std::process::Command::new(python);
                cmd.arg(&file);
                let ok = proc::run(cmd, Some(Duration::from_secs(10)), None).unwrap().ok();
                assert_eq!(ok, should_pass, "{} with {body:?}", p.name);
            }
        }
    }
}
