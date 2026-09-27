//! Programming languages Aegist knows about: which files are code, how each
//! language writes comments, which names are built in (so they're never
//! mistaken for invented ones), and the language's own tools for checking
//! and running a file.

use std::path::Path;

#[derive(Debug)]
pub struct Lang {
    pub name: &'static str,
    /// Syntax-highlighting / code-fence name.
    pub id: &'static str,
    pub exts: &'static [&'static str],
    /// Whole file names (Makefile, Dockerfile...).
    pub files: &'static [&'static str],
    pub line_comment: Option<&'static str>,
    pub block_comment: Option<(&'static str, &'static str)>,
    pub keywords: &'static [&'static str],
    /// Syntax check: program and arguments, `{}` is the file. Tried in order;
    /// the first whose program is installed is used.
    pub check: &'static [&'static [&'static str]],
    /// Run a single file.
    pub run: &'static [&'static [&'static str]],
}

impl PartialEq for Lang {
    fn eq(&self, other: &Lang) -> bool {
        self.name == other.name
    }
}

impl Lang {
    /// A comment containing `text`, in this language's style.
    pub fn comment(&self, text: &str) -> String {
        match (self.line_comment, self.block_comment) {
            (Some(lc), _) => text.lines().map(|l| if l.is_empty() { lc.to_string() } else { format!("{lc} {l}") }).collect::<Vec<_>>().join("\n"),
            (None, Some((a, b))) => format!("{a} {text} {b}"),
            (None, None) => text.to_string(),
        }
    }

    pub fn is_keyword(&self, word: &str) -> bool {
        self.keywords.contains(&word) || COMMON_WORDS.contains(&word)
    }
}

const PY_KW: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield", "self", "cls", "match", "case", "print", "len", "range", "int", "str", "float", "bool", "list", "dict", "set",
    "tuple", "open", "enumerate", "zip", "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs", "round", "isinstance",
    "issubclass", "hasattr", "getattr", "setattr", "super", "object", "type", "input", "any", "all", "iter", "next", "repr", "format",
    "hash", "id", "chr", "ord", "hex", "bin", "oct", "divmod", "pow", "bytes", "bytearray", "frozenset", "property", "staticmethod",
    "classmethod", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError", "StopIteration", "NotImplementedError",
    "AttributeError", "ZeroDivisionError", "OSError", "FileNotFoundError", "append", "extend", "insert", "pop", "remove", "keys", "values",
    "items", "get", "update", "join", "split", "strip", "lstrip", "rstrip", "replace", "startswith", "endswith", "lower", "upper", "find",
    "count", "index", "sort", "copy", "clear", "add", "discard", "read", "write", "close", "readlines", "__init__", "__name__", "__main__",
    "__str__", "__repr__", "__len__", "__eq__", "__iter__", "__next__", "__enter__", "__exit__", "__call__", "__getitem__", "__setitem__",
];
const JS_KW: &[&str] = &[
    "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "export", "extends", "finally",
    "for", "function", "if", "import", "in", "instanceof", "let", "new", "return", "super", "switch", "this", "throw", "try", "typeof",
    "var", "void", "while", "with", "yield", "async", "await", "of", "static", "get", "set", "null", "undefined", "true", "false", "NaN",
    "Infinity", "console", "log", "error", "warn", "window", "document", "Math", "JSON", "Object", "Array", "String", "Number", "Boolean",
    "Promise", "Map", "Set", "Date", "RegExp", "Error", "Symbol", "parseInt", "parseFloat", "require", "module", "exports", "process",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "requestAnimationFrame", "fetch", "length", "push", "pop", "shift",
    "unshift", "slice", "splice", "map", "filter", "reduce", "forEach", "find", "findIndex", "includes", "indexOf", "join", "split",
    "keys", "values", "entries", "assign", "freeze", "stringify", "parse", "then", "catch", "resolve", "reject", "floor", "ceil", "round",
    "random", "max", "min", "abs", "sqrt", "PI", "addEventListener", "getElementById", "querySelector", "querySelectorAll", "createElement",
    "appendChild", "getContext", "fillRect", "fillStyle", "innerHTML", "textContent", "style", "toString", "sort", "concat", "trim",
    "replace", "toUpperCase", "toLowerCase", "startsWith", "endsWith", "charAt", "substring", "has", "delete", "size", "prototype",
    "constructor", "interface", "type", "enum", "implements", "private", "public", "protected", "readonly", "any", "unknown", "never",
    "number", "string", "boolean", "void", "keyof", "as", "from", "declare", "namespace", "abstract",
];
const RS_KW: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false", "fn", "for", "if", "impl",
    "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
    "true", "type", "unsafe", "use", "where", "while", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize", "f32", "f64", "bool", "char", "str", "String", "Vec", "Option", "Some", "None", "Result", "Ok", "Err", "Box", "Rc", "Arc",
    "HashMap", "HashSet", "BTreeMap", "BTreeSet", "VecDeque", "println", "print", "eprintln", "format", "vec", "panic", "assert",
    "assert_eq", "assert_ne", "unwrap", "expect", "clone", "iter", "into_iter", "iter_mut", "collect", "map", "filter", "len", "push",
    "pop", "insert", "remove", "get", "contains", "is_empty", "to_string", "as_str", "unwrap_or", "Default", "default", "new", "from",
    "into", "Debug", "Clone", "Copy", "PartialEq", "Eq", "Hash", "Display", "fmt", "main", "derive", "std", "sum", "enumerate", "zip",
    "rev", "sort", "chars", "bytes", "lines", "split", "trim", "parse", "min", "max", "abs", "write", "writeln",
];
const GO_KW: &[&str] = &[
    "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
    "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var", "nil", "true", "false", "int",
    "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "string", "byte", "rune",
    "bool", "error", "len", "cap", "make", "new", "append", "copy", "delete", "panic", "recover", "print", "println", "fmt", "Println",
    "Printf", "Sprintf", "Errorf", "main", "os", "strings", "strconv", "errors", "Error",
];
const C_KW: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "bool", "true", "false", "NULL", "nullptr", "size_t", "printf", "scanf", "malloc",
    "calloc", "realloc", "free", "memcpy", "memset", "strlen", "strcpy", "strcmp", "fopen", "fclose", "fprintf", "sprintf", "snprintf",
    "puts", "putchar", "getchar", "exit", "main", "include", "define", "ifdef", "ifndef", "endif", "stdio", "stdlib", "string", "class",
    "public", "private", "protected", "virtual", "override", "template", "typename", "namespace", "using", "new", "delete", "this",
    "operator", "std", "cout", "cin", "endl", "vector", "map", "set", "unordered_map", "make_unique", "make_shared", "unique_ptr",
    "shared_ptr", "push_back", "size", "begin", "end", "auto", "try", "catch", "throw", "explicit", "friend", "mutable", "constexpr",
];
const JAVA_KW: &[&str] = &[
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for", "if", "implements", "import", "instanceof", "int", "interface", "long",
    "native", "new", "package", "private", "protected", "public", "return", "short", "static", "super", "switch", "synchronized", "this",
    "throw", "throws", "try", "void", "volatile", "while", "var", "true", "false", "null", "String", "System", "out", "println", "print",
    "Integer", "List", "ArrayList", "Map", "HashMap", "Override", "main", "length", "size", "add", "get", "put", "equals", "toString",
];
const SH_KW: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "in", "function", "return", "exit", "echo",
    "printf", "read", "local", "export", "set", "unset", "shift", "cd", "ls", "cat", "grep", "sed", "awk", "test", "true", "false",
];
const NONE: &[&str] = &[];

/// Words that are never evidence of an invented name in any language.
const COMMON_WORDS: &[&str] = &["TODO", "FIXME", "NOTE", "http", "https", "www", "com", "org"];

macro_rules! lang {
    ($name:expr, $id:expr, [$($e:expr),*], [$($f:expr),*], $lc:expr, $bc:expr, $kw:expr, [$($check:expr),*], [$($run:expr),*]) => {
        Lang { name: $name, id: $id, exts: &[$($e),*], files: &[$($f),*], line_comment: $lc, block_comment: $bc, keywords: $kw,
               check: &[$($check),*], run: &[$($run),*] }
    };
}

pub static LANGS: &[Lang] = &[
    lang!("Python", "python", ["py", "pyw", "pyi"], [], Some("#"), None, PY_KW,
          [&["python3", "-c", "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])", "{}"],
           &["python", "-c", "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])", "{}"]],
          [&["python3", "{}"], &["python", "{}"]]),
    lang!("JavaScript", "javascript", ["js", "mjs", "cjs", "jsx"], [], Some("//"), Some(("/*", "*/")), JS_KW,
          [&["node", "--check", "{}"]], [&["node", "{}"]]),
    lang!("TypeScript", "typescript", ["ts", "tsx", "mts", "cts"], [], Some("//"), Some(("/*", "*/")), JS_KW,
          [], [&["npx", "--yes", "tsx", "{}"]]),
    lang!("Rust", "rust", ["rs"], [], Some("//"), Some(("/*", "*/")), RS_KW,
          [&["rustc", "--edition", "2021", "--crate-type", "lib", "--emit", "metadata", "-o", "{out}", "{}"]],
          [&["rustc", "--edition", "2021", "-O", "-o", "{out}", "{}", "&&", "{out}"]]),
    lang!("Go", "go", ["go"], [], Some("//"), Some(("/*", "*/")), GO_KW, [&["gofmt", "-e", "{}"]], [&["go", "run", "{}"]]),
    lang!("C", "c", ["c", "h"], [], Some("//"), Some(("/*", "*/")), C_KW,
          [&["cc", "-fsyntax-only", "{}"], &["gcc", "-fsyntax-only", "{}"], &["clang", "-fsyntax-only", "{}"]],
          [&["cc", "-O2", "-o", "{out}", "{}", "-lm", "&&", "{out}"]]),
    lang!("C++", "cpp", ["cpp", "cc", "cxx", "hpp", "hh", "hxx"], [], Some("//"), Some(("/*", "*/")), C_KW,
          [&["c++", "-std=c++20", "-fsyntax-only", "{}"], &["g++", "-std=c++20", "-fsyntax-only", "{}"], &["clang++", "-std=c++20", "-fsyntax-only", "{}"]],
          [&["c++", "-std=c++20", "-O2", "-o", "{out}", "{}", "&&", "{out}"]]),
    lang!("C#", "cs", ["cs"], [], Some("//"), Some(("/*", "*/")), JAVA_KW, [], []),
    lang!("Java", "java", ["java"], [], Some("//"), Some(("/*", "*/")), JAVA_KW, [], [&["java", "{}"]]),
    lang!("Kotlin", "kotlin", ["kt", "kts"], [], Some("//"), Some(("/*", "*/")), JAVA_KW, [], []),
    lang!("Swift", "swift", ["swift"], [], Some("//"), Some(("/*", "*/")), NONE, [], [&["swift", "{}"]]),
    lang!("Scala", "scala", ["scala"], [], Some("//"), Some(("/*", "*/")), JAVA_KW, [], []),
    lang!("Ruby", "ruby", ["rb"], ["Rakefile", "Gemfile"], Some("#"), None, NONE, [&["ruby", "-c", "{}"]], [&["ruby", "{}"]]),
    lang!("PHP", "php", ["php"], [], Some("//"), Some(("/*", "*/")), NONE, [&["php", "-l", "{}"]], [&["php", "{}"]]),
    lang!("Lua", "lua", ["lua"], [], Some("--"), None, NONE, [&["luac", "-p", "{}"]], [&["lua", "{}"]]),
    lang!("Perl", "perl", ["pl", "pm"], [], Some("#"), None, NONE, [&["perl", "-c", "{}"]], [&["perl", "{}"]]),
    lang!("Shell", "bash", ["sh", "bash", "zsh"], [], Some("#"), None, SH_KW, [&["bash", "-n", "{}"]], [&["bash", "{}"]]),
    lang!("PowerShell", "powershell", ["ps1"], [], Some("#"), Some(("<#", "#>")), NONE, [], [&["pwsh", "-File", "{}"]]),
    lang!("Batch", "batch", ["bat", "cmd"], [], Some("REM"), None, NONE, [], []),
    lang!("HTML", "html", ["html", "htm"], [], None, Some(("<!--", "-->")), JS_KW, [], []),
    lang!("CSS", "css", ["css", "scss", "sass", "less"], [], None, Some(("/*", "*/")), NONE, [], []),
    lang!("Vue", "html", ["vue", "svelte"], [], None, Some(("<!--", "-->")), JS_KW, [], []),
    lang!("SQL", "sql", ["sql"], [], Some("--"), Some(("/*", "*/")), NONE, [], []),
    lang!("JSON", "json", ["json"], [], None, None, NONE, [], []),
    lang!("YAML", "yaml", ["yaml", "yml"], [], Some("#"), None, NONE, [], []),
    lang!("TOML", "toml", ["toml"], [], Some("#"), None, NONE, [], []),
    lang!("Markdown", "markdown", ["md", "markdown"], [], None, Some(("<!--", "-->")), NONE, [], []),
    lang!("Makefile", "makefile", ["mk"], ["Makefile", "makefile", "GNUmakefile"], Some("#"), None, NONE, [], []),
    lang!("Dockerfile", "dockerfile", ["dockerfile"], ["Dockerfile"], Some("#"), None, NONE, [], []),
    lang!("CMake", "cmake", ["cmake"], ["CMakeLists.txt"], Some("#"), None, NONE, [], []),
    lang!("Haskell", "haskell", ["hs"], [], Some("--"), Some(("{-", "-}")), NONE, [], [&["runghc", "{}"]]),
    lang!("OCaml", "ocaml", ["ml", "mli"], [], None, Some(("(*", "*)")), NONE, [], [&["ocaml", "{}"]]),
    lang!("Elixir", "elixir", ["ex", "exs"], [], Some("#"), None, NONE, [], [&["elixir", "{}"]]),
    lang!("Erlang", "erlang", ["erl"], [], Some("%"), None, NONE, [], []),
    lang!("Clojure", "clojure", ["clj", "cljs"], [], Some(";"), None, NONE, [], []),
    lang!("Lisp", "lisp", ["lisp", "el", "scm"], [], Some(";"), None, NONE, [], []),
    lang!("R", "r", ["r", "R"], [], Some("#"), None, NONE, [], [&["Rscript", "{}"]]),
    lang!("Julia", "julia", ["jl"], [], Some("#"), None, NONE, [], [&["julia", "{}"]]),
    lang!("Dart", "dart", ["dart"], [], Some("//"), Some(("/*", "*/")), NONE, [], [&["dart", "run", "{}"]]),
    lang!("Zig", "zig", ["zig"], [], Some("//"), None, NONE, [&["zig", "ast-check", "{}"]], [&["zig", "run", "{}"]]),
    lang!("Nim", "nim", ["nim"], [], Some("#"), None, NONE, [], []),
    lang!("Assembly", "asm", ["asm", "s", "S", "nasm"], [], Some(";"), None, NONE, [], []),
    lang!("Linker script", "ld", ["ld", "lds"], [], None, Some(("/*", "*/")), NONE, [], []),
    lang!("CUDA", "cpp", ["cu", "cuh"], [], Some("//"), Some(("/*", "*/")), C_KW, [], []),
    lang!("GLSL", "glsl", ["glsl", "vert", "frag", "comp"], [], Some("//"), Some(("/*", "*/")), C_KW, [], []),
    lang!("WGSL", "wgsl", ["wgsl"], [], Some("//"), Some(("/*", "*/")), NONE, [], []),
    lang!("Verilog", "verilog", ["v", "sv"], [], Some("//"), Some(("/*", "*/")), NONE, [], []),
    lang!("Protobuf", "protobuf", ["proto"], [], Some("//"), Some(("/*", "*/")), NONE, [], []),
    lang!("GraphQL", "graphql", ["graphql", "gql"], [], Some("#"), None, NONE, [], []),
    lang!("XML", "xml", ["xml", "svg", "xsd"], [], None, Some(("<!--", "-->")), NONE, [], []),
];

/// The language of a path, from its file name or extension.
pub fn for_path(path: &Path) -> Option<&'static Lang> {
    let name = path.file_name()?.to_str()?;
    if let Some(l) = LANGS.iter().find(|l| l.files.contains(&name)) {
        return Some(l);
    }
    let ext = path.extension()?.to_str()?;
    LANGS.iter().find(|l| l.exts.contains(&ext)).or_else(|| {
        let lower = ext.to_ascii_lowercase();
        LANGS.iter().find(|l| l.exts.contains(&lower.as_str()))
    })
}

/// A language by its name, id or extension ("python", "py", "C++", "cpp"...).
pub fn by_name(word: &str) -> Option<&'static Lang> {
    let w = word.trim().trim_matches(|c: char| c == '.' || c == ',').to_ascii_lowercase();
    let alias = match w.as_str() {
        "js" | "node" | "nodejs" | "node.js" => "javascript",
        "ts" => "typescript",
        "py" | "python3" => "python",
        "c++" | "cplusplus" => "cpp",
        "golang" => "go",
        "sh" | "shell" | "bash" | "zsh" => "bash",
        "rs" => "rust",
        "c#" | "csharp" => "cs",
        "web" | "webpage" | "website" | "page" => "html",
        other => other,
    };
    LANGS.iter().find(|l| l.id == alias || l.name.to_ascii_lowercase() == alias || l.exts.contains(&alias))
}

/// Whether a file is source code Aegist trains on and reads.
pub fn is_code(path: &Path) -> bool {
    for_path(path).is_some()
}

/// Every identifier-like word in `code` (letters, digits, underscores, not
/// starting with a digit), in order, with repeats.
pub fn identifiers(code: &str) -> impl Iterator<Item = &str> {
    let bytes = code.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                return Some(&code[start..i]);
            } else if c.is_ascii_digit() {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_languages_by_path_and_name() {
        assert_eq!(for_path(Path::new("src/main.rs")).unwrap().name, "Rust");
        assert_eq!(for_path(Path::new("a/Makefile")).unwrap().name, "Makefile");
        assert_eq!(for_path(Path::new("x.PY")).unwrap().name, "Python");
        assert!(for_path(Path::new("notes.txt")).is_none());
        assert_eq!(by_name("js").unwrap().name, "JavaScript");
        assert_eq!(by_name("C++").unwrap().name, "C++");
        assert_eq!(by_name("website").unwrap().name, "HTML");
        assert!(by_name("klingon").is_none());
    }

    #[test]
    fn comments_and_identifiers() {
        let py = by_name("python").unwrap();
        assert_eq!(py.comment("hi\nthere"), "# hi\n# there");
        assert_eq!(by_name("css").unwrap().comment("x"), "/* x */");
        let ids: Vec<&str> = identifiers("let x_1 = foo(2abc, _bar) + 3.5e2;").collect();
        assert_eq!(ids, vec!["let", "x_1", "foo", "_bar"]);
        assert!(py.is_keyword("self") && py.is_keyword("TODO") && !py.is_keyword("frobnicate"));
    }
}
