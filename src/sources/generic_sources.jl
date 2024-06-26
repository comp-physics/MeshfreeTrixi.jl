"""
    SourceTerms(; kwargs...) = SourceTerms(NamedTuple(kwargs))

Convenience wrapper for generic source terms. Expects sources 
to be passed in as a NamedTuple or its entries.
"""
struct SourceTerms
    sources::NamedTuple
end

Base.iterate(terms::SourceTerms, state...) = iterate(terms.sources, state...)
Base.length(terms::SourceTerms) = length(terms.sources)
Base.collect(terms::SourceTerms) = collect(terms.sources)
Base.keys(terms::SourceTerms) = keys(terms.sources)
Base.values(terms::SourceTerms) = values(terms.sources)
Base.pairs(terms::SourceTerms) = pairs(terms.sources)

SourceTerms(; kwargs...) = SourceTerms(NamedTuple(kwargs))

function Base.show(io::IO, terms::SourceTerms)
    type_names = [String(nameof(typeof(source))) for (_, source) in pairs(terms.sources)]
    print(io, join(type_names, ", "))
end

include("hyperviscosity.jl")
include("IGR.jl")
