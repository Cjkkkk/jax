# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

XLA_COMMIT = "71e3a0d40d7f368b239dc664e3bb0a455a80541b"
XLA_SHA256 = "5729c77cc32c5513f18802dd46dcf4c5b0457a0fb5ad29d1c543734d23afdaba"

def repo():
    tf_http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)),
    )

    # For development, one often wants to make changes to the TF repository as well
    # as the JAX repository. You can override the pinned repository above with a
    # local checkout by either:
    # a) overriding the TF repository on the build.py command line by passing a flag
    #    like:
    #    python build/build.py build --local_xla_path=/path/to/xla
    #    or
    # b) by commenting out the http_archive above and uncommenting the following:
    # local_repository(
    #    name = "xla",
    #    path = "/path/to/xla",
    # )
