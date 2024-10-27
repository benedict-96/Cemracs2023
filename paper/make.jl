using GeometricMachineLearning
using Documenter
using DocumenterCitations
# using Weave

const output_type = isempty(ARGS) ? :cemracs : ARGS[2] == "arxiv" ? :arxiv : :cemracs

# this is necessary to avoid warnings. See https://documenter.juliadocs.org/dev/man/syntax/
ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "VolumePreservingTransformer.bib"))

const latex_format = Documenter.LaTeX()

# if platform is set to "none" then no output pdf is generated
const latex_format_no_pdf = Documenter.LaTeX(platform = "none")

const format = isempty(ARGS) ? latex_format : ARGS[1] == "no_pdf" ? latex_format_no_pdf : latex_format

cemracs_finish = [  "Rigid Body" => "rigid_body.md",
                    "References" => "references.md"]

arxiv_finish = [    "References" => "references.md",
                    "Rigid Body" => "rigid_body.md"]

pages_in_common = [
    "Introduction" => "introduction.md",
    "Divergence-Free Vector Fields" => "divergence_free_vector_fields.md",
    "Transformer" => "transformer.md",
    "VPFF" => "volume_preserving_feed_forward.md",
    "VPT" => "volume_preserving_transformer.md",
    "Results" => "results.md",
    "Conclusion" => "conclusion.md",
    "Acknowledgements" => "acknowledgements.md",
]

pages = output_type == :cemracs ? vcat(pages_in_common, cemracs_finish) : vcat(pages_in_common, arxiv_finish)

makedocs(;
    plugins = [bib],
    sitename = "Volume-Preserving Neural Network Architectures",
    authors = "Michael Kraus, Benedikt Brantner",
    format = format,
    pages = pages,
)