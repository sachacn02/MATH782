import nox


@nox.session(python=["3.13"])
def tests(session):
    session.install(".[dev]")
    session.run("pytest")


@nox.session
def lint(session):
    session.install("isort", "black")
    session.run("isort", "--check", ".")
    session.run("black", "--check", ".")
