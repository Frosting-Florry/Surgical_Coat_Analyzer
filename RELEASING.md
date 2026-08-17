# Windows EXE release checklist

1. Run the regression tests:

   ```powershell
   python -m unittest discover -s tests -v
   ```

2. Build the executable on Windows:

   ```powershell
   .\build_windows.bat
   ```

3. Verify `dist\Surgical_Coat_Analyzer.exe` on a Windows computer without the source environment.

4. Record a SHA-256 checksum:

   ```powershell
   Get-FileHash -Algorithm SHA256 .\dist\Surgical_Coat_Analyzer.exe
   ```

5. Create a version tag and GitHub Release, then attach the EXE and include its checksum in the release notes. The EXE is intentionally excluded from the Git repository by `.gitignore`.

