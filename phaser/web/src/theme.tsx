import { AppShell, AppShellStylesNames, createTheme, Tabs, TabsStylesNames } from "@mantine/core"
import * as tabs_classes from './Tabs.module.css'
import * as appshell_classes from './AppShell.module.css'

console.log(`tabs_classes: ${JSON.stringify(tabs_classes)}`);

export const makeTheme = () => createTheme({
    //fontFamily: 'Open Sans, sans-serif',
    components: {
        Tabs: Tabs.extend({
            classNames: tabs_classes as Partial<Record<TabsStylesNames, string>>,
        }),
        AppShell: AppShell.extend({
            classNames: appshell_classes as Partial<Record<AppShellStylesNames, string>>,
        }),
    },
    defaultRadius: 'md',

    primaryColor: "blue",
    colors: {
        dark: [
            '#e2daeb',  // text color
            '#b8b8b8',
            '#828282',
            '#696969',  // placeholder, disabled color
            '#424242',  // border
            '#3b3b3b',  // hover
            '#2e2e2e',  // disabled
            '#251f1f',  // bg
            '#1f1f1f',  // dark filled
            '#141414',  // dark filled hover
        ],
    },
});

export const cssVariableResolver = (theme: ReturnType<typeof makeTheme>) => ({
    variables: {
    },
    light: {
    },
    dark: {
    },
});