using ContinuousDPs
using Documenter

DocMeta.setdocmeta!(ContinuousDPs, :DocTestSetup, :(using ContinuousDPs); recursive=true)

makedocs(;
    modules=[ContinuousDPs],
    authors="QuantEcon",
    sitename="ContinuousDPs.jl",
    format=Documenter.HTML(;
        canonical="https://QuantEcon.github.io/ContinuousDPs.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:none,  # Don't error on missing docs
    warnonly=[:missing_docs, :docs_block]  # Convert errors to warnings
)

deploydocs(;
    repo="github.com/QuantEcon/ContinuousDPs.jl",
    devbranch="master",
)