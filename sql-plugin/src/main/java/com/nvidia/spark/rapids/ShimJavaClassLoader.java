/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids;

public class ShimJavaClassLoader  extends ClassLoader {
  private final String shimPrefix;

  public ShimJavaClassLoader(ClassLoader parent, String shimPrefix) {
    super(parent);
    this.shimPrefix = shimPrefix;
  }

  private Class<?> loadClassInternal(String classFileName, String className) throws ClassNotFoundException {
    final java.io.InputStream inputStream = getParent().getResourceAsStream(classFileName);
    if (inputStream == null) return null;
    try {
      System.out.println("GERA_DEBUG: reading bytes from : " + classFileName);
      final byte[] classBytes = org.apache.commons.io.IOUtils.toByteArray(inputStream);
      return defineClass(className, classBytes, 0, classBytes.length);
    } catch (java.io.IOException e) {
      throw new ClassNotFoundException("CNF because of IO", e);
    }
  }

  @Override
  protected Class<?> findClass(String className) throws ClassNotFoundException {
    final String originalClassFileName = className
        .replace(".", "/") + ".class";
    final String shimmedClassFileName = shimPrefix + "/" + originalClassFileName;
    final Class<?> shimmedClass = loadClassInternal(shimmedClassFileName, className);
    if (shimmedClass != null) {
      System.out.println("GERA_DEBUG: loaded shimmed class" + className +
          " from " + shimmedClassFileName);
      return shimmedClass;
    }
    if (className.startsWith("com.nvidia")) {
      final Class<?> unshimmedClass = loadClassInternal(originalClassFileName, className);
      if (unshimmedClass != null) {
        System.out.println("GERA_DEBUG: loaded UNSHIMMED class" + className +
            " from " + originalClassFileName);
        return unshimmedClass;
      }
    }
    System.out.println("GERA_DEBUG delegating to parent " + className);
    return getParent().loadClass(className);
  }

  @Override
  public java.net.URL getResource(String originalFileName) {
    final String resourceFileName = shimPrefix + "/" + originalFileName;
    final java.net.URL shimmedResourceUrl = getParent().getResource(resourceFileName);
    if (shimmedResourceUrl != null) return shimmedResourceUrl;
    final java.net.URL unshimmedResourceURL = getParent().getResource(originalFileName);
    if (unshimmedResourceURL != null) return unshimmedResourceURL;

    return getParent().getResource(originalFileName);
  }
}
