using ContinuousDPs
using BasisMatrices
using Documenter

DocMeta.setdocmeta!(ContinuousDPs, :DocTestSetup, :(using ContinuousDPs); recursive=true)

# Sync README.md to src/index.md for the documentation
function sync_readme_to_index()
    src = joinpath(@__DIR__, "..", "README.md")
    dst = joinpath(@__DIR__, "src", "index.md")
    text = read(src, String)
    write(dst, text)
end

sync_readme_to_index()

makedocs(;
    modules=[ContinuousDPs],
    authors="QuantEcon",
    sitename="ContinuousDPs.jl",
    format=Documenter.HTML(;
        canonical="https://QuantEcon.github.io/ContinuousDPs.jl",
        edit_link="main",
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
    devbranch="main",
)