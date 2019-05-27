import React, { useState, useEffect } from 'react'
import './App.css'
import InputBase from '@material-ui/core/InputBase'
import { makeStyles } from '@material-ui/core/styles'
import AppBar from '@material-ui/core/AppBar'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import Button from '@material-ui/core/Button'
import Avatar from '@material-ui/core/Avatar'
import Switch from '@material-ui/core/Switch'
import List from '@material-ui/core/List'
import IconButton from '@material-ui/core/IconButton'
import MenuIcon from '@material-ui/icons/Menu'
import Drawer from '@material-ui/core/Drawer'
import data from './data'
import gen from 'random-seed'
import shuffle from 'shuffle-seed'

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
  },
  menuButton: {
    // marginRight: theme.spacing(2),
    right: '-12px',
  },
  title: {
    flexGrow: 1,
  },
  drawer: {
    width: '400px',
    flexShrink: 0,
  },
  drawerPaper: {
    width: '400px',
  },
}))

const avatarStyles = {
  width: '30px',
  height: '30px',
  color: 'black',
  fontWeight: 'bold',
  margin: '0 10px',
}

function App() {
  const [seed, setSeed] = useState('')
  const [open, updateOpen] = useState(false)
  const [isSpyMaster, updateSpyMaster] = useState(false)
  const words = shuffle.shuffle(data.original, seed).slice(0, 25)
  const classes = useStyles()
  const random = gen.create(seed)
  const first = random(2) % 2 ? 'red' : 'blue'
  const colors = shuffle.shuffle(
    [
      ...Array(8).fill('red'),
      ...Array(8).fill('blue'),
      ...Array(7).fill('yellow'),
      'black',
      first,
    ],
    seed,
  )
  const generatedDataInitialState = words.map((word, i) => ({
    word,
    color: colors[i],
    selected: false,
  }))

  const [generatedData, updateGeneratedData] = useState(
    generatedDataInitialState,
  )

  useEffect(_ => updateGeneratedData(generatedDataInitialState), [seed])
  const [redScore, blueScore] = generatedData.reduce(
    (acc, { color, selected }) => {
      if (selected && color === 'red') acc[0]--
      if (selected && color === 'blue') acc[1]--
      return acc
    },
    first === 'red' ? [9, 8] : [8, 9],
  )

  return (
    <div className='App'>
      <div className='appBar'>
        <AppBar position='static'>
          <Toolbar variant='dense'>
            <Typography variant='h6'>KodeNames</Typography>
            <div className={classes.title} />
            <InputBase
              placeholder='Seed...'
              classes={{
                root: classes.inputRoot,
                input: classes.inputInput,
              }}
              onChange={e => setSeed(e.target.value)}
            />
            <Avatar style={{ ...avatarStyles, backgroundColor: '#ff0000' }}>
              {redScore}
            </Avatar>
            <Avatar style={{ ...avatarStyles, backgroundColor: '#00eeee' }}>
              {blueScore}
            </Avatar>
            <IconButton
              edge='start'
              className={classes.menuButton}
              color='inherit'
              aria-label='Menu'
              onClick={() => updateOpen(true)}
            >
              <MenuIcon />
            </IconButton>
          </Toolbar>
        </AppBar>
      </div>
      <table className='table'>
        <tbody>
          {new Array(5).fill(1).map((_, i) => (
            <tr key={`row-${i}`}>
              {new Array(5).fill(1).map((_, j) => {
                const cardData = generatedData[i * 5 + j]
                return (
                  <Card
                    isSpyMaster={isSpyMaster}
                    key={cardData.word}
                    {...cardData}
                    onClick={() => {
                      cardData.selected = !cardData.selected
                      updateGeneratedData([...generatedData])
                    }}
                  />
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <Drawer
        className={classes.drawer}
        variant='persistent'
        anchor='right'
        open={open}
        classes={{
          paper: classes.drawerPaper,
        }}
      >
        Spy Master
        <Switch
          checked={isSpyMaster}
          onChange={() => updateSpyMaster(!isSpyMaster)}
        />
        <List onClick={() => updateOpen(false)}>list 1</List>
        <List>list 2</List>
      </Drawer>
    </div>
  )
}

export default App

function Card({ word, selected, color, onClick, isSpyMaster }) {
  const selection = isSpyMaster
    ? `${selected ? ' green' : ` ${color}`}`
    : `${selected ? ` ${color}` : ''}`
  return (
    <td className={`card${selection}`} onClick={onClick}>
      {word}
    </td>
  )
}