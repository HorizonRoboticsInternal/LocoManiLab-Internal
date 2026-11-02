import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'g1_backend',
  tagline: 'Documentation for sim2sim, sim2real, and teleoperation workflows',

  url: 'https://locomanilab.github.io',
  baseUrl: '/g1_backend/',

  organizationName: 'LocoManiLab',
  projectName: 'g1_backend',

  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '.',
          exclude: [
            'README.md',
            'getting-started/README.md',
            'tutorials/README.md',
            'i18n/**',
            'package.json',
            'package-lock.json',
            'tsconfig.json',
            'sidebars.ts',
            'docusaurus.config.ts',
            'src/**',
            'static/**',
            'build/**',
            '.docusaurus/**',
            'node_modules/**',
          ],
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          editUrl: 'https://github.com/LocoManiLab/g1_backend/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    navbar: {
      title: 'g1_backend',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/LocoManiLab/g1_backend',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/getting-started/overview'},
            {label: 'Tutorials', to: '/tutorials/offline-motion-tracking'},
            {label: 'Reference', to: '/reference/tracking-framework'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: 'https://github.com/LocoManiLab/g1_backend'},
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} Gamma Robotics and sim2real Contributors. MIT License. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'yaml', 'python'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
