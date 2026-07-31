#ifndef VANTAGE_REACTIONS_INLINE_H
#define VANTAGE_REACTIONS_INLINE_H
// Empty in compiled mode (definitions live in exactly one .cpp);
// `inline` in header-only mode (definitions are included into every TU).
#ifdef VANTAGE_REACTIONS_HEADER_ONLY
#define VANTAGE_REACTIONS_INLINE inline
#else
#define VANTAGE_REACTIONS_INLINE
#endif
#endif
