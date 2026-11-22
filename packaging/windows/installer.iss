; Inno Setup script for Kol Studio (Windows 10/11, 64-bit).
; Build: iscc packaging\windows\installer.iss /DAppVersion=0.1.0
; Input:  dist\KolStudio\  (PyInstaller folder build)   Output: dist\KolStudio-<version>-windows-x64-setup.exe
; Models are not inside the installer: the app downloads them on first run.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{8F3A2C6E-6B1D-4E7A-9C2F-4B5D7E1A9C30}
AppName=Kol Studio
AppVerName=Kol Studio {#AppVersion}
AppVersion={#AppVersion}
AppPublisher=Kol Studio
DefaultDirName={autopf}\KolStudio
DefaultGroupName=Kol Studio
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=KolStudio-{#AppVersion}-windows-x64-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\KolStudio.exe
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=..\..\LICENSE

[Languages]
Name: "hebrew"; MessagesFile: "compiler:Languages\Hebrew.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\KolStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Kol Studio"; Filename: "{app}\KolStudio.exe"
Name: "{group}\{cm:UninstallProgram,Kol Studio}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Kol Studio"; Filename: "{app}\KolStudio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\KolStudio.exe"; Description: "{cm:LaunchProgram,Kol Studio}"; Flags: nowait postinstall skipifsilent

; Downloaded models, voices and settings live in %APPDATA%\KolStudio and are kept on uninstall.
