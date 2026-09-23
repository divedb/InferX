#!/bin/sh
# Records exactly how the checked-in Boost subset was produced.
# Beast 361 (third_party/beast, post-1.89 develop) requires Boost develop
# headers; this is the minimal header-only closure, pinned to the develop
# commit SHAs below (resolved 2026-09-22). The include/ tree is checked in;
# re-run only after reviewing the diff and refreshing the SHAs.
set -e
cd "$(dirname "$0")"
mkdir -p include/boost
curl -sL "https://github.com/boostorg/align/archive/440281d63d1c0b7c7fde63ded67b4860b57d5756.tar.gz" | tar -xz -C include/boost --strip-components=3 "align-440281d63d1c0b7c7fde63ded67b4860b57d5756/include/boost"
curl -sL "https://github.com/boostorg/asio/archive/a7dc25b4cb6c49a6946d86ea20664f1027203225.tar.gz" | tar -xz -C include/boost --strip-components=3 "asio-a7dc25b4cb6c49a6946d86ea20664f1027203225/include/boost"
curl -sL "https://github.com/boostorg/assert/archive/612d35df55376c6412f0477ed62d4d241f8861b7.tar.gz" | tar -xz -C include/boost --strip-components=3 "assert-612d35df55376c6412f0477ed62d4d241f8861b7/include/boost"
curl -sL "https://github.com/boostorg/bind/archive/8cc29fc19db49e791743c821821e201b46ab9c66.tar.gz" | tar -xz -C include/boost --strip-components=3 "bind-8cc29fc19db49e791743c821821e201b46ab9c66/include/boost"
curl -sL "https://github.com/boostorg/config/archive/1a75a3e2ed18a49b3257fa0ce93cde590d1efac4.tar.gz" | tar -xz -C include/boost --strip-components=3 "config-1a75a3e2ed18a49b3257fa0ce93cde590d1efac4/include/boost"
curl -sL "https://github.com/boostorg/container_hash/archive/2698b43803c012601e6bb1a6116e83767b97986c.tar.gz" | tar -xz -C include/boost --strip-components=3 "container_hash-2698b43803c012601e6bb1a6116e83767b97986c/include/boost"
curl -sL "https://github.com/boostorg/core/archive/c434b91503c97da219d7987cba89f8f85f0b4a85.tar.gz" | tar -xz -C include/boost --strip-components=3 "core-c434b91503c97da219d7987cba89f8f85f0b4a85/include/boost"
curl -sL "https://github.com/boostorg/describe/archive/5e7b4b84c8d105093687b940a45ac22df47b1ab4.tar.gz" | tar -xz -C include/boost --strip-components=3 "describe-5e7b4b84c8d105093687b940a45ac22df47b1ab4/include/boost"
curl -sL "https://github.com/boostorg/function/archive/18650af5175ea247aebc60ff12db1b477123d5dc.tar.gz" | tar -xz -C include/boost --strip-components=3 "function-18650af5175ea247aebc60ff12db1b477123d5dc/include/boost"
curl -sL "https://github.com/boostorg/intrusive/archive/69abbe445685b8a311db925e581bf4a770686752.tar.gz" | tar -xz -C include/boost --strip-components=3 "intrusive-69abbe445685b8a311db925e581bf4a770686752/include/boost"
curl -sL "https://github.com/boostorg/io/archive/5c738076aa9aafc7392b2004c6b6f268b076a9f0.tar.gz" | tar -xz -C include/boost --strip-components=3 "io-5c738076aa9aafc7392b2004c6b6f268b076a9f0/include/boost"
curl -sL "https://github.com/boostorg/logic/archive/ebff10a43bf6512548eae231a1b9c81260dd98dc.tar.gz" | tar -xz -C include/boost --strip-components=3 "logic-ebff10a43bf6512548eae231a1b9c81260dd98dc/include/boost"
curl -sL "https://github.com/boostorg/move/archive/ee8a65dcc6e1f2a232d6f6dbb8a9230916715b35.tar.gz" | tar -xz -C include/boost --strip-components=3 "move-ee8a65dcc6e1f2a232d6f6dbb8a9230916715b35/include/boost"
curl -sL "https://github.com/boostorg/mp11/archive/0daffd6401724ddf473fd39d28f68e138d9a1303.tar.gz" | tar -xz -C include/boost --strip-components=3 "mp11-0daffd6401724ddf473fd39d28f68e138d9a1303/include/boost"
curl -sL "https://github.com/boostorg/optional/archive/f2212618a5679713fb8770be14d74a915a5170e5.tar.gz" | tar -xz -C include/boost --strip-components=3 "optional-f2212618a5679713fb8770be14d74a915a5170e5/include/boost"
curl -sL "https://github.com/boostorg/predef/archive/761ee22750b11dc813617051c5be9d6a5dfadb2b.tar.gz" | tar -xz -C include/boost --strip-components=3 "predef-761ee22750b11dc813617051c5be9d6a5dfadb2b/include/boost"
curl -sL "https://github.com/boostorg/scope_exit/archive/5450074da7f1d338eb5774b245bd7a255f92680e.tar.gz" | tar -xz -C include/boost --strip-components=3 "scope_exit-5450074da7f1d338eb5774b245bd7a255f92680e/include/boost"
curl -sL "https://github.com/boostorg/smart_ptr/archive/6e945160d788b8efdfc49ba4af1f8797cacd7c97.tar.gz" | tar -xz -C include/boost --strip-components=3 "smart_ptr-6e945160d788b8efdfc49ba4af1f8797cacd7c97/include/boost"
curl -sL "https://github.com/boostorg/static_string/archive/2ff9d3535e853f4913e85e7fea28f84b04ea5d81.tar.gz" | tar -xz -C include/boost --strip-components=3 "static_string-2ff9d3535e853f4913e85e7fea28f84b04ea5d81/include/boost"
curl -sL "https://github.com/boostorg/system/archive/bc7c00fa67501ceadfde8e920835502340e8b899.tar.gz" | tar -xz -C include/boost --strip-components=3 "system-bc7c00fa67501ceadfde8e920835502340e8b899/include/boost"
curl -sL "https://github.com/boostorg/throw_exception/archive/0924b53b40d1da33301f94fb97518f5a7df31e9b.tar.gz" | tar -xz -C include/boost --strip-components=3 "throw_exception-0924b53b40d1da33301f94fb97518f5a7df31e9b/include/boost"
curl -sL "https://github.com/boostorg/type_traits/archive/e6275ccf01c9cf8775ef0cb6188bd58f3b167a0f.tar.gz" | tar -xz -C include/boost --strip-components=3 "type_traits-e6275ccf01c9cf8775ef0cb6188bd58f3b167a0f/include/boost"
curl -sL "https://github.com/boostorg/utility/archive/30a5eca1002aeae22f95d7b1e824290255d1c663.tar.gz" | tar -xz -C include/boost --strip-components=3 "utility-30a5eca1002aeae22f95d7b1e824290255d1c663/include/boost"
curl -sL "https://github.com/boostorg/variant2/archive/fa7d53b5cd1bda93e93e11d7974881ccabdb945a.tar.gz" | tar -xz -C include/boost --strip-components=3 "variant2-fa7d53b5cd1bda93e93e11d7974881ccabdb945a/include/boost"
curl -sL "https://github.com/boostorg/winapi/archive/094612ee33a8dce960cd075537266fdb4788d059.tar.gz" | tar -xz -C include/boost --strip-components=3 "winapi-094612ee33a8dce960cd075537266fdb4788d059/include/boost"
