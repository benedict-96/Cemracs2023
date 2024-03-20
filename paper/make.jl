using GeometricMachineLearning
using Documenter
using DocumenterCitations
# using Weave

# this is necessary to avoid warnings. See https://documenter.juliadocs.org/dev/man/syntax/
ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "VolumePreservingTransformer.bib"))

const latex_format = Documenter.LaTeX()

# if platform is set to "none" then no output pdf is generated
const latex_format_no_pdf = Documenter.LaTeX(platform = "none")

const format = isempty(ARGS) ? latex_format : ARGS[1] == "no_pdf" ? latex_format_no_pdf : latex_format

makedocs(;
    plugins = [bib],
    sitename = "Volume-Preserving Neural Network Architectures",
    authors = "Michael Kraus, Benedikt Brantner",
    format = format,
    pages=[
        "Introduction" => "introduction.md",
        "Rigid Body" => "rigid_body.md",
        "Transformer" => "transformer.md",
        "VPFF" => "volume_preserving_feed_forward.md",
        "VPT" => "volume_preserving_transformer.md",
        "References" => "references.md",
    ],
)